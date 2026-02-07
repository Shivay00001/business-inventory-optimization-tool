from fastapi import FastAPI
from pydantic import BaseModel
from src.optimization.optimizer import InventoryOptimizer

app = FastAPI(title="Inventory Optimization API")

class SKUData(BaseModel):
    sku_id: str
    annual_demand: float
    avg_demand: float
    std_dev_demand: float
    avg_lead_time: float
    order_cost: float
    holding_cost: float

@app.post("/optimize")
async def optimize_inventory(data: SKUData):
    results = InventoryOptimizer.optimize_sku(data.dict())
    return {
        "sku_id": data.sku_id,
        "recommendations": results
    }

@app.get("/health")
async def health():
    return {"status": "ok", "engine": "InventoryOptimizer_v1"}
