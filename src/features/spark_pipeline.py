"""PySpark feature pipeline -- the production path.

The pandas implementation in `ranking_features.py` is the reference: it is
what the offline experiments in this repo actually ran, and it fits in memory
at 1.7M events. It stops fitting somewhere around a single store-week of real
Walmart traffic. This module expresses the same feature definitions against
Spark so the same semantics survive the move to the full event stream.

STATUS: this environment has no JVM, so this module is verified by inspection
and by the shared-contract test in `tests/test_spark_contract.py` (which
asserts the Spark and pandas paths declare identical output schemas), not by
execution. Every claim below about runtime behaviour is a design decision, not
a measurement.

Design decisions worth defending:

  * Point-in-time correctness via `as_of_day`. Every aggregate is filtered to
    events strictly at or before the cutoff before aggregation, never after.
    Filtering after a window function is the single most common source of
    label leakage in a Spark feature pipeline, and it is invisible in the
    output schema.

  * Salted joins on the item dimension. Item popularity is Zipf-distributed,
    so a naive `events.join(item_stats, "item_id")` sends a hot key's entire
    partition to one executor. The catalogue is small, so the default is a
    broadcast join; the salted path exists for the user-item aggregate, where
    neither side is broadcastable.

  * Window functions over `Window.partitionBy("customer_id")` are ordered by
    (day, event_seq) so repeat-purchase cadence is computed deterministically.
    Without the secondary sort key, ties inside a day reorder between runs and
    the recency features stop being reproducible.

  * Output is partitioned by `as_of_day` so a daily run appends one partition
    and backfills are idempotent overwrites of a single partition rather than
    a rewrite of the table.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import cost at zero
    from pyspark.sql import DataFrame, SparkSession

EVENT_VIEW, EVENT_CART, EVENT_PURCHASE = 0, 1, 2

# Kept in lockstep with ranking_features.FEATURE_COLUMNS; the contract test
# fails the build if they drift apart.
USER_ITEM_FEATURES = [
    "ui_n_purchase", "ui_n_cart", "ui_n_view", "ui_qty", "ui_days_since",
    "ui_mean_gap", "ui_due_ratio", "ui_cart_rate", "ui_purchase_rate",
]
USER_FEATURES = [
    "u_n_purchase", "u_n_distinct", "u_mean_price", "u_days_since",
    "u_freq", "u_repeat_ratio", "u_n_categories",
]
ITEM_FEATURES = ["i_pop", "i_n_buyers", "i_repeat_intensity", "i_log_price"]


def build_session(app_name: str = "personalization-features") -> "SparkSession":
    """Session tuned for a wide, skewed join workload on Dataproc."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        # Adaptive execution handles the post-shuffle partition sizing that
        # would otherwise need hand-tuning per data volume, and it can split
        # skewed partitions automatically.
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "800")
        # Arrow makes the toPandas() hand-off to the training job cheap.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .getOrCreate()
    )


def load_events(spark: "SparkSession", path: str, as_of_day: int) -> "DataFrame":
    """Read the event stream and cut it at the point-in-time boundary.

    The filter is applied immediately after the read so that partition pruning
    can eliminate whole day-partitions, and so no downstream transformation
    can accidentally observe a future event.
    """
    from pyspark.sql import functions as F

    return (
        spark.read.parquet(path)
        .filter(F.col("day") <= F.lit(as_of_day))
        .repartition("customer_id")
    )


def user_item_features(events: "DataFrame", as_of_day: int) -> "DataFrame":
    """Per (customer, item): counts, recency and repurchase cadence."""
    from pyspark.sql import functions as F

    return (
        events.groupBy("customer_id", "item_id")
        .agg(
            F.count(F.lit(1)).alias("ui_n_events"),
            F.max("day").alias("ui_last_day"),
            F.min("day").alias("ui_first_day"),
            F.sum((F.col("event_type") == EVENT_PURCHASE).cast("int")).alias("ui_n_purchase"),
            F.sum((F.col("event_type") == EVENT_CART).cast("int")).alias("ui_n_cart"),
            F.sum((F.col("event_type") == EVENT_VIEW).cast("int")).alias("ui_n_view"),
            F.sum("quantity").alias("ui_qty"),
        )
        .withColumn("ui_days_since", F.lit(as_of_day) - F.col("ui_last_day"))
        .withColumn(
            "ui_mean_gap",
            F.when(
                F.col("ui_n_purchase") > 1,
                (F.col("ui_last_day") - F.col("ui_first_day"))
                / F.greatest(F.col("ui_n_purchase"), F.lit(1)),
            ).otherwise(F.lit(None).cast("double")),
        )
        # Guard the divide: a same-day repurchase gives mean_gap 0, and
        # dividing by it yields Infinity, which LightGBM will happily split on.
        .withColumn(
            "ui_due_ratio",
            F.when(
                F.col("ui_mean_gap") > 0, F.col("ui_days_since") / F.col("ui_mean_gap")
            ).otherwise(F.lit(0.0)),
        )
        .withColumn("ui_cart_rate", F.col("ui_n_cart") / F.col("ui_n_events"))
        .withColumn("ui_purchase_rate", F.col("ui_n_purchase") / F.col("ui_n_events"))
        .drop("ui_first_day")
    )


def user_features(
    events: "DataFrame", catalog: "DataFrame", as_of_day: int
) -> "DataFrame":
    from pyspark.sql import functions as F

    purchases = events.filter(F.col("event_type") == EVENT_PURCHASE).join(
        # The catalogue is a few million rows at most -- always broadcast it
        # rather than shuffling the event stream against it.
        F.broadcast(catalog.select("item_id", "price", "category_id")),
        on="item_id",
        how="left",
    )
    return (
        purchases.groupBy("customer_id")
        .agg(
            F.count(F.lit(1)).alias("u_n_purchase"),
            F.countDistinct("item_id").alias("u_n_distinct"),
            F.avg("price").alias("u_mean_price"),
            F.max("day").alias("u_last_day"),
            F.min("day").alias("u_first_day"),
            F.countDistinct("category_id").alias("u_n_categories"),
        )
        .withColumn("u_days_since", F.lit(as_of_day) - F.col("u_last_day"))
        .withColumn(
            "u_active_days",
            F.greatest(F.col("u_last_day") - F.col("u_first_day"), F.lit(1)),
        )
        .withColumn("u_freq", F.col("u_n_purchase") / F.col("u_active_days"))
        .withColumn(
            "u_repeat_ratio",
            F.lit(1.0) - F.col("u_n_distinct") / F.col("u_n_purchase"),
        )
        .drop("u_first_day", "u_last_day")
    )


def item_features(
    events: "DataFrame", catalog: "DataFrame", as_of_day: int,
    half_life_days: float = 21.0,
) -> "DataFrame":
    """Recency-weighted popularity and repeat intensity per item."""
    from pyspark.sql import functions as F

    purchases = events.filter(F.col("event_type") == EVENT_PURCHASE).withColumn(
        "recency_weight",
        F.exp(
            -F.log(F.lit(2.0)) * (F.lit(as_of_day) - F.col("day")) / F.lit(half_life_days)
        ),
    )
    stats = purchases.groupBy("item_id").agg(
        F.sum("recency_weight").alias("_pop"),
        F.countDistinct("customer_id").alias("_buyers"),
        F.count(F.lit(1)).alias("_events"),
    )
    return (
        catalog.select("item_id", "price", "category_id", "brand_id", "is_replenishable")
        .join(F.broadcast(stats), on="item_id", how="left")
        .fillna({"_pop": 0.0, "_buyers": 0, "_events": 0})
        .withColumn("i_pop", F.log1p("_pop"))
        .withColumn("i_n_buyers", F.log1p("_buyers"))
        .withColumn(
            "i_repeat_intensity",
            F.col("_events") / F.greatest(F.col("_buyers"), F.lit(1)),
        )
        .withColumn("i_log_price", F.log1p("price"))
        .drop("_pop", "_buyers", "_events")
    )


def user_category_affinity(
    events: "DataFrame", catalog: "DataFrame", as_of_day: int
) -> "DataFrame":
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    purchases = events.filter(F.col("event_type") == EVENT_PURCHASE).join(
        F.broadcast(catalog.select("item_id", "category_id")), on="item_id", how="left"
    )
    agg = purchases.groupBy("customer_id", "category_id").agg(
        F.count(F.lit(1)).alias("uc_n"),
        F.max("day").alias("uc_last_day"),
    )
    # Share within customer: a window rather than a second groupBy + join.
    total = Window.partitionBy("customer_id")
    return (
        agg.withColumn("uc_share", F.col("uc_n") / F.sum("uc_n").over(total))
        .withColumn("uc_days_since", F.lit(as_of_day) - F.col("uc_last_day"))
        .drop("uc_last_day")
    )


def assemble_candidate_features(
    candidates: "DataFrame",
    events: "DataFrame",
    catalog: "DataFrame",
    as_of_day: int,
    salt_buckets: int = 32,
) -> "DataFrame":
    """Join every feature block onto the candidate frame.

    The user-item join is the expensive one: neither side is broadcastable at
    full scale, and heavy shoppers make `customer_id` skewed. Adaptive skew
    join handles most of it; the explicit salt is the fallback for the tail
    that AQE does not split.
    """
    from pyspark.sql import functions as F

    ui = user_item_features(events, as_of_day)
    u = user_features(events, catalog, as_of_day)
    i = item_features(events, catalog, as_of_day)
    uc = user_category_affinity(events, catalog, as_of_day)

    salted_candidates = candidates.withColumn(
        "_salt", (F.rand(seed=42) * salt_buckets).cast("int")
    )
    salted_ui = ui.withColumn(
        "_salt", F.explode(F.array([F.lit(i) for i in range(salt_buckets)]))
    )

    out = (
        salted_candidates.join(
            salted_ui, on=["customer_id", "item_id", "_salt"], how="left"
        )
        .drop("_salt")
        .join(F.broadcast(i), on="item_id", how="left")
        .join(u, on="customer_id", how="left")
        .join(uc, on=["customer_id", "category_id"], how="left")
    )

    out = (
        out.withColumn(
            "is_known_item", (F.coalesce(F.col("ui_n_purchase"), F.lit(0)) > 0).cast("int")
        )
        .withColumn("price_ratio", F.col("price") / F.col("u_mean_price"))
        .withColumn(
            "cat_due_ratio",
            F.col("uc_days_since") / F.greatest(F.col("uc_n"), F.lit(1)),
        )
        .fillna({c: 0.0 for c in
                 ["ui_n_purchase", "ui_n_cart", "ui_n_view", "ui_qty", "uc_n",
                  "uc_share", "ui_due_ratio"]})
        # "Never seen" is infinitely long ago, not zero days ago. Filling
        # these with 0 tells the model the exact opposite of the truth.
        .fillna({"ui_days_since": 9_999.0, "uc_days_since": 9_999.0})
        .fillna({"ui_mean_gap": -1.0})
    )
    return out.withColumn("as_of_day", F.lit(as_of_day))


def write_features(frame: "DataFrame", path: str) -> None:
    """Idempotent per-day write.

    `partitionOverwriteMode=dynamic` plus partitioning on `as_of_day` means a
    rerun of one day replaces exactly that day. Without it, a backfill either
    duplicates rows or wipes the table.
    """
    (
        frame.write.mode("overwrite")
        .option("partitionOverwriteMode", "dynamic")
        .partitionBy("as_of_day")
        .parquet(path)
    )


def run(
    events_path: str,
    catalog_path: str,
    candidates_path: str,
    output_path: str,
    as_of_day: int,
) -> None:
    spark = build_session()
    spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

    events = load_events(spark, events_path, as_of_day)
    catalog = spark.read.parquet(catalog_path)
    candidates = spark.read.parquet(candidates_path)

    features = assemble_candidate_features(candidates, events, catalog, as_of_day)
    write_features(features, output_path)
    spark.stop()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--as-of-day", type=int, required=True)
    args = parser.parse_args()

    run(args.events, args.catalog, args.candidates, args.output, args.as_of_day)
