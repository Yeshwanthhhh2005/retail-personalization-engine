"""Simulated Walmart-shaped retail event stream.

Real customer data cannot ship with an assignment, so we simulate one with the
properties that actually make retail personalization hard:

  * Zipf-skewed item popularity  -> a popularity baseline is genuinely strong.
  * Heavy repeat purchase        -> "already bought" is signal, not leakage.
  * Latent taste structure       -> there is real per-customer signal to learn.
  * Category seasonality         -> the last week is not the same as week one.
  * A view -> cart -> purchase funnel with decreasing volume.

The latent taste vectors are held out of the modelling feature set, so a model
has to recover them from behaviour. That gives us a defensible ceiling to
measure the learned models against.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG, DataConfig

CATEGORY_NAMES = [
    "Fresh Produce", "Dairy & Eggs", "Bakery", "Meat & Seafood", "Frozen Foods",
    "Pantry Staples", "Snacks", "Beverages", "Coffee & Tea", "Baby Care",
    "Household Cleaning", "Paper Goods", "Personal Care", "Beauty", "Pharmacy",
    "Vitamins", "Pet Supplies", "Apparel Men", "Apparel Women", "Apparel Kids",
    "Footwear", "Home Decor", "Kitchen & Dining", "Bedding & Bath", "Furniture",
    "Electronics", "Mobile & Accessories", "Computers", "TV & Audio", "Gaming",
    "Toys", "Sports & Outdoors", "Auto Care", "Tools & Hardware", "Garden",
    "Office Supplies", "Books & Media", "Party Supplies", "Seasonal", "Jewelry",
]

# Categories bought on a weekly rhythm. Drives the repeat-purchase machinery.
REPLENISHABLE_CATEGORIES = frozenset(range(0, 17))

EVENT_VIEW, EVENT_CART, EVENT_PURCHASE = 0, 1, 2


def build_catalog(
    cfg: DataConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    """Item master plus the hidden per-item taste vector."""
    n = cfg.n_items
    category_id = rng.integers(0, cfg.n_categories, size=n)
    brand_id = rng.integers(0, cfg.n_brands, size=n)

    # Price is log-normal, centred per category so Electronics != Produce.
    cat_price_center = rng.uniform(0.6, 4.2, size=cfg.n_categories)
    price = np.exp(cat_price_center[category_id] + rng.normal(0, 0.55, size=n))
    price = np.clip(price, 0.75, 2400.0).round(2)

    # Zipf popularity: the head item outsells the tail by ~3 orders of magnitude.
    ranks = rng.permutation(n) + 1
    popularity = 1.0 / np.power(ranks, cfg.popularity_zipf)
    popularity /= popularity.sum()

    # Taste vectors correlate within a category: someone who likes one organic
    # yoghurt tends to like the others.
    cat_vec = rng.normal(0, 1.0, size=(cfg.n_categories, cfg.latent_dim))
    item_vec = 0.75 * cat_vec[category_id] + 0.65 * rng.normal(
        0, 1, size=(n, cfg.latent_dim)
    )
    item_vec /= np.linalg.norm(item_vec, axis=1, keepdims=True)

    catalog = pd.DataFrame({
        "item_id": np.arange(n, dtype=np.int32),
        "category_id": category_id.astype(np.int16),
        "category_name": [CATEGORY_NAMES[c] for c in category_id],
        "brand_id": brand_id.astype(np.int16),
        "price": price.astype(np.float32),
        "is_replenishable": np.isin(category_id, list(REPLENISHABLE_CATEGORIES)),
        "base_popularity": popularity.astype(np.float32),
    })
    return catalog, item_vec


def build_customers(
    cfg: DataConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, np.ndarray]:
    n = cfg.n_customers
    customer_vec = rng.normal(0, 1, size=(n, cfg.latent_dim))
    customer_vec /= np.linalg.norm(customer_vec, axis=1, keepdims=True)

    customers = pd.DataFrame({
        "customer_id": np.arange(n, dtype=np.int32),
        "home_store_id": rng.integers(0, cfg.n_stores, size=n).astype(np.int16),
        # 0 = indifferent to price, 1 = strongly prefers cheap.
        "price_sensitivity": rng.beta(2.2, 2.2, size=n).astype(np.float32),
        "tenure_days": rng.integers(5, 1500, size=n).astype(np.int16),
        "prefers_online": rng.random(n) < 0.46,
    })
    return customers, customer_vec


def build_affinity_pools(
    cfg: DataConfig,
    customer_vec: np.ndarray,
    item_vec: np.ndarray,
    catalog: pd.DataFrame,
    price_sensitivity: np.ndarray,
    chunk: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-customer candidate pool and its sampling distribution.

    Materialising the full customer x item score matrix is 20k x 4k here, and
    hundreds of millions x millions at Walmart scale, so we chunk it and keep
    only a top-N pool per customer. That mirrors production: you never score
    the whole catalogue for every user.
    """
    log_price = np.log(catalog["price"].to_numpy())
    z_price = (log_price - log_price.mean()) / log_price.std()
    log_pop = np.log(catalog["base_popularity"].to_numpy() + 1e-12)
    z_pop = (log_pop - log_pop.mean()) / log_pop.std()

    pool_size = cfg.pool_size
    pools = np.zeros((cfg.n_customers, pool_size), dtype=np.int32)
    probs = np.zeros((cfg.n_customers, pool_size), dtype=np.float32)

    for start in range(0, cfg.n_customers, chunk):
        end = min(start + chunk, cfg.n_customers)
        score = cfg.taste_weight * (customer_vec[start:end] @ item_vec.T)
        score += cfg.popularity_weight * z_pop[None, :]
        score -= cfg.price_weight * price_sensitivity[start:end, None] * z_price[None, :]

        idx = np.argpartition(-score, pool_size, axis=1)[:, :pool_size]
        row = np.arange(end - start)[:, None]
        top = score[row, idx]
        top -= top.max(axis=1, keepdims=True)
        p = np.exp(top / cfg.pool_temperature)
        p /= p.sum(axis=1, keepdims=True)

        pools[start:end] = idx.astype(np.int32)
        probs[start:end] = p.astype(np.float32)
        del score
    return pools, probs


def build_seasonality(cfg: DataConfig, rng: np.random.Generator) -> np.ndarray:
    """(n_days, n_categories) multiplicative demand factor."""
    days = np.arange(cfg.n_days)
    phase = rng.uniform(0, 2 * np.pi, size=cfg.n_categories)
    amp = rng.uniform(0.05, 0.45, size=cfg.n_categories)
    season = 1.0 + amp[None, :] * np.sin(
        2 * np.pi * days[:, None] / 30.0 + phase[None, :]
    )
    weekend = np.where((days % 7) >= 5, 1.22, 1.0)  # catalogue-wide uplift
    return (season * weekend[:, None]).astype(np.float32)


def _weighted_draw(
    cum: np.ndarray, rng: np.random.Generator, size: int
) -> np.ndarray:
    """Categorical sampling by inverse CDF -- far cheaper than rng.choice."""
    return np.searchsorted(cum, rng.random(size) * cum[-1])


def generate_events(cfg: DataConfig | None = None) -> dict[str, pd.DataFrame]:
    """Generate the catalog, customer master and interaction log."""
    cfg = cfg or CONFIG.data
    rng = np.random.default_rng(cfg.seed)

    catalog, item_vec = build_catalog(cfg, rng)
    customers, customer_vec = build_customers(cfg, rng)

    price_sensitivity = customers["price_sensitivity"].to_numpy()
    pools, probs = build_affinity_pools(
        cfg, customer_vec, item_vec, catalog, price_sensitivity
    )
    season = build_seasonality(cfg, rng)

    item_cat = catalog["category_id"].to_numpy()
    replenishable = catalog["is_replenishable"].to_numpy()
    log_price = np.log(catalog["price"].to_numpy())
    z_price = (log_price - log_price.mean()) / log_price.std()

    # Sessions, ordered by (customer, day) so each customer's history is causal.
    n_sessions = rng.poisson(cfg.avg_sessions_per_customer, size=cfg.n_customers) + 1
    total_sessions = int(n_sessions.sum())
    session_customer = np.repeat(
        np.arange(cfg.n_customers, dtype=np.int32), n_sessions
    )
    # Mild recency uplift so the held-out final week is not starved of traffic.
    session_day = np.floor(
        cfg.n_days * rng.beta(1.15, 1.0, size=total_sessions)
    ).astype(np.int16)
    np.clip(session_day, 0, cfg.n_days - 1, out=session_day)
    order = np.lexsort((session_day, session_customer))
    session_customer, session_day = session_customer[order], session_day[order]
    session_size = 1 + rng.poisson(2.6, size=total_sessions)

    # Discovery draws (customers stumbling onto popular items) are pre-sampled
    # in bulk; drawing them one at a time dominates the runtime otherwise.
    global_pop = catalog["base_popularity"].to_numpy().astype(np.float64)
    global_pop /= global_pop.sum()
    discovery_pool = rng.choice(
        cfg.n_items, size=max(1, int(0.12 * session_size.sum())), p=global_pop
    )
    discovery_cursor = 0

    home_store = customers["home_store_id"].to_numpy()
    prefers_online = customers["prefers_online"].to_numpy()
    store_of_session = np.where(
        rng.random(total_sessions) < 0.82,
        home_store[session_customer],
        rng.integers(0, cfg.n_stores, size=total_sessions),
    ).astype(np.int16)
    online_of_session = np.where(
        rng.random(total_sessions) < 0.80,
        prefers_online[session_customer],
        rng.random(total_sessions) < 0.5,
    )

    chunks: list[dict[str, np.ndarray]] = []
    owned: dict[int, list[int]] = {}
    repeat_cut = cfg.repeat_purchase_rate
    discovery_cut = repeat_cut + 0.08

    for s in range(total_sessions):
        c = int(session_customer[s])
        d = int(session_day[s])
        k = int(session_size[s])

        # Today's pool weights: static affinity tilted by category seasonality.
        pool = pools[c]
        w = probs[c].astype(np.float64) * season[d, item_cat[pool]]
        cum_pool = np.cumsum(w)

        history = owned.get(c)
        mode = rng.random(k)
        picks = np.empty(k, dtype=np.int32)

        from_pool = mode >= discovery_cut
        n_pool = int(from_pool.sum())
        if n_pool:
            picks[from_pool] = pool[_weighted_draw(cum_pool, rng, n_pool)]

        from_disc = (mode >= repeat_cut) & (mode < discovery_cut)
        n_disc = int(from_disc.sum())
        if n_disc:
            if discovery_cursor + n_disc > len(discovery_pool):
                discovery_pool = rng.choice(
                    cfg.n_items, size=len(discovery_pool), p=global_pop
                )
                discovery_cursor = 0
            picks[from_disc] = discovery_pool[
                discovery_cursor : discovery_cursor + n_disc
            ]
            discovery_cursor += n_disc

        from_repeat = mode < repeat_cut
        n_repeat = int(from_repeat.sum())
        if n_repeat:
            if history:
                # Re-buy, biased to replenishables and to recent purchases.
                tail = np.asarray(history[-12:], dtype=np.int32)
                wt = np.where(replenishable[tail], 2.4, 1.0)
                picks[from_repeat] = tail[_weighted_draw(np.cumsum(wt), rng, n_repeat)]
            else:
                # No history yet: fall back to the affinity pool.
                picks[from_repeat] = pool[_weighted_draw(cum_pool, rng, n_repeat)]

        # Funnel. Affinity and price sensitivity drive cart and purchase rates.
        affinity = item_vec[picks] @ customer_vec[c]
        logit = 0.35 + 2.1 * affinity - 0.55 * price_sensitivity[c] * z_price[picks]
        p_cart = 1.0 / (1.0 + np.exp(-logit))
        carted = rng.random(k) < p_cart
        purchased = carted & (rng.random(k) < 0.66)

        cart_items = picks[carted]
        purchase_items = picks[purchased]
        n_p = len(purchase_items)
        quantity = (
            1
            + (rng.random(n_p) < 0.28).astype(np.int8)
            + (replenishable[purchase_items] & (rng.random(n_p) < 0.22)).astype(np.int8)
        )

        items = np.concatenate([picks, cart_items, purchase_items])
        types = np.concatenate([
            np.full(k, EVENT_VIEW, dtype=np.int8),
            np.full(len(cart_items), EVENT_CART, dtype=np.int8),
            np.full(n_p, EVENT_PURCHASE, dtype=np.int8),
        ])
        qty = np.concatenate([
            np.zeros(k + len(cart_items), dtype=np.int8),
            quantity.astype(np.int8),
        ])
        chunks.append({
            "customer_id": np.full(len(items), c, dtype=np.int32),
            "item_id": items.astype(np.int32),
            "day": np.full(len(items), d, dtype=np.int16),
            "event_type": types,
            "store_id": np.full(len(items), store_of_session[s], dtype=np.int16),
            "is_online": np.full(len(items), online_of_session[s], dtype=bool),
            "quantity": qty,
        })

        if n_p:
            owned.setdefault(c, []).extend(purchase_items.tolist())

    events = pd.DataFrame({
        col: np.concatenate([ch[col] for ch in chunks]) for col in chunks[0]
    })
    # Stable within-day ordering so temporal features are reproducible.
    events["event_seq"] = np.arange(len(events), dtype=np.int64)
    events = events.sort_values(["day", "event_seq"], kind="stable").reset_index(drop=True)
    events["revenue"] = (
        events["quantity"].to_numpy()
        * catalog["price"].to_numpy()[events["item_id"].to_numpy()]
    ).astype(np.float32)

    return {
        "events": events,
        "catalog": catalog,
        "customers": customers,
        # Generative truth. Held out of every feature set; used only to compute
        # an oracle ceiling in the evaluation report.
        "latent_customer": pd.DataFrame(customer_vec.astype(np.float32)),
        "latent_item": pd.DataFrame(item_vec.astype(np.float32)),
    }


def main() -> None:
    cfg = CONFIG.data
    tables = generate_events(cfg)
    for name, df in tables.items():
        df.to_parquet(CONFIG.data_dir / f"{name}.parquet", index=False)

    ev = tables["events"]
    purchases = ev[ev.event_type == EVENT_PURCHASE]
    distinct = purchases.groupby("customer_id").item_id.nunique().sum()
    top100 = purchases.item_id.value_counts().head(100).sum()

    print(f"events         : {len(ev):>11,}")
    print(f"  views        : {int((ev.event_type == EVENT_VIEW).sum()):>11,}")
    print(f"  carts        : {int((ev.event_type == EVENT_CART).sum()):>11,}")
    print(f"  purchases    : {len(purchases):>11,}")
    print(f"customers      : {ev.customer_id.nunique():>11,}")
    print(f"items covered  : {ev.item_id.nunique():>11,} / {cfg.n_items:,}")
    print(f"revenue        : {purchases.revenue.sum():>11,.0f}")
    print(f"repeat share   : {1 - distinct / len(purchases):>11.3f}")
    print(f"top-100 share  : {top100 / len(purchases):>11.3f}")
    print(f"written to     : {CONFIG.data_dir}")


if __name__ == "__main__":
    main()
