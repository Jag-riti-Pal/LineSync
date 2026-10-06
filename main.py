import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List

from inventory_engine import InventoryEngine
from vision_inspector import ConveyorInspector

app = FastAPI(
    title="LineSync API - FMCG Quality & Inventory Orchestration",
    version="1.1.0"
)

inventory_engine = InventoryEngine(service_level_z=1.65)
inspector = ConveyorInspector()

inspection_history: List[str] = []
CURRENT_WAREHOUSE_INVENTORY = 4200.0

class InventoryParameters(BaseModel):
    avg_daily_demand: float = 500.0
    std_daily_demand: float = 60.0
    avg_lead_time_days: float = 7.0
    std_lead_time_days: float = 1.5

@app.get("/")
def read_root():
    return {"service": "LineSync FMCG API", "status": "Online", "model": "YOLOv8n"}

@app.post("/api/reset")
def reset_inspection_history():
    global inspection_history
    inspection_history.clear()
    return {"status": "SUCCESS", "message": "Line telemetry cleared."}

@app.post("/api/inspect")
async def inspect_conveyor_item(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        raw_result = inspector.inspect_frame_bytes(contents)

        # Log pass/fail
        inspection_history.append(raw_result["status"])

        # Base64 encode the annotated image frame for frontend rendering
        b64_image = base64.b64encode(raw_result["annotated_frame_bytes"]).decode("utf-8")

        return {
            "status": raw_result["status"],
            "defect_type": raw_result["defect_type"],
            "confidence": raw_result["confidence"],
            "detected_objects": raw_result["detected_objects"],
            "annotated_image_base64": b64_image,
            "timestamp": raw_result["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/inventory-status")
def get_inventory_status(params: InventoryParameters):
    total_inspected = len(inspection_history)
    rejections = inspection_history.count("REJECT")
    
    scrap_rate = (rejections / total_inspected) if total_inspected > 0 else 0.0

    base_ss = inventory_engine.calculate_base_safety_stock(
        params.avg_daily_demand,
        params.std_daily_demand,
        params.avg_lead_time_days,
        params.std_lead_time_days
    )
    
    dynamic_ss = inventory_engine.calculate_dynamic_safety_stock(base_ss, scrap_rate)
    reorder_point = inventory_engine.calculate_reorder_point(
        params.avg_daily_demand,
        params.avg_lead_time_days,
        dynamic_ss
    )

    needs_reorder = CURRENT_WAREHOUSE_INVENTORY <= reorder_point

    return {
        "production_metrics": {
            "total_inspected_units": total_inspected,
            "total_rejected_units": rejections,
            "current_scrap_rate_pct": round(scrap_rate * 100, 2)
        },
        "inventory_optimization": {
            "base_safety_stock_units": round(base_ss, 2),
            "dynamic_safety_stock_units": dynamic_ss,
            "reorder_point_units": reorder_point,
            "current_inventory": CURRENT_WAREHOUSE_INVENTORY,
            "reorder_trigger_active": needs_reorder
        }
    }