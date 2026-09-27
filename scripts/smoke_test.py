#!/usr/bin/env python3
"""
Smoke test for the Inventory Optimization API.

Boots the app in-process (no server needed), hits the main endpoints and
asserts 200s. Exits non-zero on any failure.

Usage:
    API_KEY=dev-api-key-change-me python3 scripts/smoke_test.py
    # or simply: python3 scripts/smoke_test.py   (defaults to dev key)

Keep this script in the repo — CI / pre-deploy gate.
"""
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.getenv("SMOKE_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-api-key-change-me")

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def req(method, path, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main():
    good = {"X-API-Key": API_KEY}

    # 1. /health is public, returns 200
    code, body = req("GET", "/health")
    check("GET /health -> 200", code == 200, f"got {code}")
    if code == 200:
        h = json.loads(body)
        check("/health payload shape", h.get("status") == "ok", str(h))

    # 2. Dashboard is public, returns HTML
    code, body = req("GET", "/")
    check("GET / (dashboard) -> 200", code == 200, f"got {code}")
    check("dashboard is HTML", b"<html" in body.lower(), "no <html> marker")

    # 3. Protected endpoints reject missing key with 401 JSON
    code, body = req("POST", "/optimize", {"sku_id": "X", "annual_demand": 1,
                                           "avg_demand": 1, "std_dev_demand": 0,
                                           "avg_lead_time": 1, "order_cost": 1,
                                           "holding_cost": 1})
    check("POST /optimize without key -> 401", code == 401, f"got {code}")
    check("401 is structured JSON, no stack trace", b"traceback" not in body.lower(),
          body[:80].decode(errors="replace"))

    sku = {
        "sku_id": "RICE-5KG",
        "annual_demand": 12000,
        "avg_demand": 33,
        "std_dev_demand": 30,
        "avg_lead_time": 7,
        "order_cost": 100,
        "holding_cost": 5,
    }

    # 4. /optimize with key -> 200 and sane numbers
    code, body = req("POST", "/optimize", sku, good)
    check("POST /optimize with key -> 200", code == 200, f"got {code}")
    if code == 200:
        r = json.loads(body)
        eoq_expected = ((2 * 12000 * 100) / 5) ** 0.5  # 692.82
        check("EOQ math correct", abs(r["eoq"] - eoq_expected) < 0.01,
              f"eoq={r.get('eoq')}")
        check("reorder_point > 0", r.get("reorder_point", 0) > 0)

    # 5. Invalid input -> 422 structured error, no stack trace
    bad = dict(sku, annual_demand=-5)
    code, body = req("POST", "/optimize", bad, good)
    check("POST /optimize invalid input -> 422", code == 422, f"got {code}")
    check("422 is structured JSON, no stack trace", b"traceback" not in body.lower(),
          body[:80].decode(errors="replace"))

    # 6. Batch endpoint
    code, body = req("POST", "/optimize/batch", {"skus": [sku, {**sku, "sku_id": "OIL-1L"}]}, good)
    check("POST /optimize/batch -> 200", code == 200, f"got {code}")
    if code == 200:
        check("batch returns 2 results", len(json.loads(body)["results"]) == 2)

    # 7. CSV export
    code, body = req("POST", "/export/csv", {"skus": [sku]}, good)
    check("POST /export/csv -> 200", code == 200, f"got {code}")
    if code == 200:
        text = body.decode()
        check("CSV has header + row",
              text.splitlines()[0].startswith("sku_id,") and "RICE-5KG" in text,
              text.splitlines()[0] if text else "")

    # 8. Wrong key -> 401
    code, _ = req("POST", "/optimize", sku, {"X-API-Key": "wrong-key"})
    check("POST /optimize wrong key -> 401", code == 401, f"got {code}")

    print()
    if FAILURES:
        print(f"SMOKE TEST FAILED: {len(FAILURES)} failure(s): {FAILURES}")
        return 1
    print("SMOKE TEST PASSED: all checks green.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
