"""Exercise the service in-process and report real latency percentiles."""
from __future__ import annotations

import time

import numpy as np
from fastapi.testclient import TestClient

from src.serve.app import app


def main() -> None:
    t0 = time.perf_counter()
    with TestClient(app) as client:
        print(f"startup (model + feature store load): {time.perf_counter() - t0:.1f}s")
        print("health:", client.get("/health").json())

        rng = np.random.default_rng(0)
        customers = rng.integers(0, 20_000, size=60)

        first = client.post("/recommend", json={"customer_id": int(customers[0]), "k": 5})
        body = first.json()
        print(f"\ncustomer {body['customer_id']}  cold_start={body['cold_start']}")
        for r in body["recommendations"]:
            print(f"  item {r['item_id']:>4}  {r['category']:<22} "
                  f"${r['price']:>7.2f}  score {r['score']:>7.3f}  "
                  f"sources={','.join(r['sources'])}")

        cold = 0
        for c in customers:
            resp = client.post("/recommend", json={"customer_id": int(c), "k": 10})
            assert resp.status_code == 200, resp.text
            cold += resp.json()["cold_start"]

        print(f"\n{len(customers)} requests, {cold} cold-start")
        print("metrics:", client.get("/metrics").json())

        bad = client.post("/recommend", json={"customer_id": 999_999, "k": 5})
        print("unknown customer ->", bad.status_code, bad.json()["detail"])


if __name__ == "__main__":
    main()
