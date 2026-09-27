# Business Inventory Optimization Tool

Real EOQ / safety-stock / reorder-point math over HTTP for Indian SMBs —
kirana chains, pharmacies, spare-part shops. Includes a **single-page
dashboard** (no code needed) and a **CSV export** (opens in Excel/Sheets) so a
non-technical shop owner can actually use it.

- `POST /optimize` — one SKU → EOQ, safety stock, reorder point, orders/year
- `POST /optimize/batch` — up to 500 SKUs in one call
- `POST /export/csv` — the same results as a downloadable CSV report
- `GET /` — browser dashboard (enter API key, fill form or paste CSV rows)
- `GET /health` — public health check (200)

## Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `API_KEY` | **Yes** (runtime) | `dev-api-key-change-me` | Shared secret — send as the `X-API-Key` header on `/optimize`, `/optimize/batch`, `/export/csv`. Change before any real use. |
| `PORT` | No | `8000` | Port to listen on (Render sets this automatically). |
| `CORS_ALLOWED_ORIGINS` | No | _(empty = no CORS)_ | Comma-separated origins allowed to call the API from browsers, e.g. `https://your-shop.example.com`. Credentials are never allowed. |
| `LOG_LEVEL` | No | `info` | `debug` / `info` / `warning` / `error` |

Copy `.env.example` to `.env` and fill in `API_KEY`. Never commit `.env` (it is
git-ignored).

## Local run

```bash
# 1. Clone and enter
git clone https://github.com/Shivay00001/business-inventory-optimization-tool.git
cd business-inventory-optimization-tool

# 2. Create env file
cp .env.example .env
# edit .env and set a real API_KEY

# 3. Install deps and run
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000

# 4. Verify (in another terminal)
python3 scripts/smoke_test.py   # boots nothing — hits the running server
```

Or with Docker Compose:

```bash
cp .env.example .env   # set API_KEY
docker compose up --build
```

Try it:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/optimize \
  -H 'Content-Type: application/json' -H 'X-API-Key: <your-key>' \
  -d '{"sku_id":"RICE-5KG","annual_demand":12000,"avg_demand":33,"std_dev_demand":30,"avg_lead_time":7,"order_cost":100,"holding_cost":5}'
```

Or open `http://localhost:8000/` in a browser — the built-in dashboard.

## Deploy on Render (free tier)

1. Push this repo to GitHub (or use the existing repo).
2. Render dashboard → **New + → Web Service** → connect the repo.
3. Render auto-detects `render.yaml` (see below); if not: Build command
   `pip install -r requirements.txt`, Start command
   `uvicorn src.main:app --host 0.0.0.0 --port $PORT`.
4. **Environment → Add `API_KEY`** with a long random value (required).
5. Deploy. Health check path: `/health`.

## Auth

All endpoints except `GET /health` and `GET /` (dashboard) require the
`X-API-Key` header matching the `API_KEY` env var. Missing/wrong key → `401`
with a JSON body, never a stack trace.

## Errors

All errors are structured JSON — validation failures return `422` with
per-field messages, auth failures `401`, unexpected failures `500`
`{"error":"internal_error", ...}`. Stack traces are logged server-side only.

## Tests

```bash
pytest tests/ -q                    # unit tests (13)
python3 scripts/smoke_test.py       # live-endpoint smoke test (needs server up)
```

## Notes

- Math is real (EOQ = √(2DS/H), safety stock = z·√L·σ) — no stubs, no mocks.
- `main.go` / `index.html` at the repo root are legacy scaffolds not used by
  the Python API; the sellable product is the FastAPI service + dashboard.
- `analytics.py` / `forecast.py` under `src/optimization/` are unexposed
  library modules (ABC analysis, demand forecasting) — available for a
  follow-up paid add-on (e.g. a `/forecast` endpoint), not wired into routes.
