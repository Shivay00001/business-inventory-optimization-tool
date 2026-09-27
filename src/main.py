"""
Inventory Optimization API — production entry point.

Exposes EOQ / safety-stock / reorder-point math over HTTP with
API-key auth, structured JSON errors, and a built-in dashboard.

Env vars (see .env.example):
    API_KEY               Shared secret required in the X-API-Key header
                          (all endpoints except /health and the dashboard)
    PORT                  Port to listen on (default: 8000)
    CORS_ALLOWED_ORIGINS  Comma-separated origins, e.g.
                          https://app.example.com (default: none = no CORS)
    LOG_LEVEL             Logging level (default: info)
"""
import csv
import io
import logging
import os
import time
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ for local dev; ignored in prod

from src.optimization.optimizer import InventoryOptimizer

logger = logging.getLogger("inventory-api")

# ---------------------------------------------------------------------------
# Configuration (all from env — no hardcoded secrets/URLs/credentials)
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "dev-api-key-change-me")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
_allowed = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
CORS_ALLOWED_ORIGINS: List[str] = (
    [o.strip() for o in _allowed.split(",") if o.strip()] if _allowed else []
)

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
if API_KEY == "dev-api-key-change-me":
    logger.warning(
        "Running with the default dev API_KEY. Set a real API_KEY env var before any real deployment."
    )

app = FastAPI(
    title="Inventory Optimization API",
    description="EOQ / safety-stock / reorder-point optimization for SMB inventory (kirana chains, pharmacies, spare-part shops).",
    version="1.1.0",
)

# Sane CORS: only the explicitly configured origins, credentials OFF,
# and CORS only matters for browser callers — the built-in dashboard is same-origin.
if CORS_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

START_TIME = time.time()


# ---------------------------------------------------------------------------
# Structured errors — never leak stack traces to clients
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "message": "One or more request fields are invalid.",
            "details": [
                {"field": ".".join(str(p) for p in err["loc"]), "message": err["msg"]}
                for err in exc.errors()
            ],
        },
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, exc: Exception):
    logger.exception("Unhandled error serving request: %s", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An unexpected error occurred."},
    )


# ---------------------------------------------------------------------------
# Auth: shared-secret header on all non-public endpoints
# ---------------------------------------------------------------------------
def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    if x_api_key != API_KEY:
        raise _unauthorized()


def _unauthorized() -> Exception:
    from fastapi import HTTPException

    return HTTPException(
        status_code=401,
        detail={"error": "unauthorized", "message": "Valid X-API-Key header required."},
    )


# ---------------------------------------------------------------------------
# Models — all external inputs validated at the boundary
# ---------------------------------------------------------------------------
class SKUData(BaseModel):
    sku_id: str = Field(..., min_length=1, max_length=128)
    annual_demand: float = Field(..., gt=0, description="Units sold per year")
    avg_demand: float = Field(..., gt=0, description="Average daily demand")
    std_dev_demand: float = Field(..., ge=0, description="Std dev of daily demand")
    avg_lead_time: float = Field(..., gt=0, description="Average lead time (days)")
    order_cost: float = Field(..., gt=0, description="Cost per order (₹)")
    holding_cost: float = Field(..., gt=0, description="Holding cost per unit per year (₹)")
    service_level: float = Field(0.95, gt=0, lt=1, description="Target service level 0-1")


class BatchRequest(BaseModel):
    skus: List[SKUData] = Field(..., min_length=1, max_length=500)


# ---------------------------------------------------------------------------
# Core computation helper (shared by JSON, CSV, and dashboard paths)
# ---------------------------------------------------------------------------
def _recommendation(data: SKUData) -> dict:
    payload = data.model_dump()
    res = InventoryOptimizer.optimize_sku(
        {
            "annual_demand": payload["annual_demand"],
            "order_cost": payload["order_cost"],
            "holding_cost": payload["holding_cost"],
            "avg_lead_time": payload["avg_lead_time"],
            "std_dev_demand": payload["std_dev_demand"],
            "avg_demand": payload["avg_demand"],
        }
    )
    # Service level is honoured via a corrected safety stock, not the
    # optimizer's default 0.95.
    ss = float(
        InventoryOptimizer.calculate_safety_stock(
            payload["avg_lead_time"], payload["std_dev_demand"], payload["service_level"]
        )
    )
    rop = float(
        InventoryOptimizer.calculate_reorder_point(
            payload["avg_demand"], payload["avg_lead_time"], ss
        )
    )
    return {
        "sku_id": data.sku_id,
        "eoq": round(float(res["eoq"]), 2),
        "safety_stock": round(ss, 2),
        "reorder_point": round(rop, 2),
        "orders_per_year": round(payload["annual_demand"] / res["eoq"], 2)
        if res["eoq"] > 0
        else 0.0,
    }


# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "InventoryOptimizer_v1",
        "uptime_seconds": int(time.time() - START_TIME),
    }


@app.get("/", include_in_schema=False)
async def dashboard():
    """Single-page dashboard — a non-technical SMB owner can run this in a browser."""
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "dashboard.html"),
        media_type="text/html",
    )


# ---------------------------------------------------------------------------
# Protected endpoints
# ---------------------------------------------------------------------------
@app.post("/optimize", dependencies=[Depends(require_api_key)])
async def optimize_inventory(data: SKUData):
    """Optimize a single SKU. Returns EOQ, safety stock, reorder point."""
    return _recommendation(data)


@app.post("/optimize/batch", dependencies=[Depends(require_api_key)])
async def optimize_batch(batch: BatchRequest):
    """Optimize up to 500 SKUs in one call."""
    return {"count": len(batch.skus), "results": [_recommendation(s) for s in batch.skus]}


@app.post("/export/csv", dependencies=[Depends(require_api_key)])
async def export_csv(batch: BatchRequest):
    """Optimization report as CSV — the deliverable a shop owner can open in Excel/Sheets."""
    results = [_recommendation(s) for s in batch.skus]
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=["sku_id", "eoq", "safety_stock", "reorder_point", "orders_per_year"],
    )
    writer.writeheader()
    writer.writerows(results)
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=inventory-optimization.csv"},
    )
