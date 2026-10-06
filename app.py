import streamlit as st
import requests
import numpy as np
import cv2
import time
import pandas as pd
import os
import glob
import random
import base64

st.set_page_config(
    page_title="LineSync | FMCG Operations Console",
    page_icon="🏭",
    layout="wide"
)

API_BASE_URL = "http://127.0.0.1:8000"

st.title("🏭 LineSync: FMCG Quality & Inventory Control")
st.caption("Edge Computer Vision Inspection coupled with Prescriptive Replenishment Logic")
st.markdown("---")

tab_sim, tab_manual = st.tabs(["🚀 Automated Batch Simulator", "📹 Single Frame Inspection"])

ACCEPT_DIR = "data/good"
REJECT_DIR = "data/bad"

# =========================================================
# TAB 1: AUTOMATED BATCH SIMULATOR
# =========================================================
with tab_sim:
    st.subheader("High-Speed Conveyor Telemetry Simulation")
    st.write("Simulate an operational run of units passing camera sensors to evaluate dynamic buffer response.")

    sim_col1, sim_col2 = st.columns([1, 2], gap="large")

    with sim_col1:
        st.markdown("#### Run Parameters")
        batch_size = st.slider("Batch Size (Units)", min_value=20, max_value=120, value=50, step=10)
        expected_defect_prob = st.slider("Injected Defect Probability (%)", min_value=0, max_value=30, value=8, step=1) / 100.0
        stream_delay = st.slider("Line Speed Delay (sec/unit)", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

        start_sim_btn = st.button("▶ Run Conveyor Simulation", type="primary", use_container_width=True)
        reset_btn = st.button("🔄 Reset Line Memory", use_container_width=True)

    with sim_col2:
        st.markdown("#### Live Telemetry & Stock Reaction")
        prog_bar = st.progress(0)
        status_text = st.empty()
        
        col_m1, col_m2, col_m3 = st.columns(3)
        metric_units = col_m1.empty()
        metric_scrap = col_m2.empty()
        metric_dyn_ss = col_m3.empty()

        chart_placeholder = st.empty()
        alert_placeholder = st.empty()

    if reset_btn:
        try:
            requests.post(f"{API_BASE_URL}/api/reset")
            st.success("Conveyor telemetry cleared.")
        except Exception as e:
            st.error(f"Failed to reset: {e}")

    if start_sim_btn:
        # 1. Reset backend before starting the batch
        try:
            requests.post(f"{API_BASE_URL}/api/reset")
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach FastAPI server. Start Uvicorn on http://127.0.0.1:8000.")
            st.stop()

        chart_data = []

        # Gather dataset image lists
        reject_images = glob.glob(f"{REJECT_DIR}/*.*") if os.path.exists(REJECT_DIR) else []
        accept_images = glob.glob(f"{ACCEPT_DIR}/*.*") if os.path.exists(ACCEPT_DIR) else []

        # 2. Sequential Unit Processing Loop
        for i in range(1, batch_size + 1):
            is_defective = np.random.rand() < expected_defect_prob
            file_bytes = None
            filename = f"unit_{i}.jpg"

            # Check if dataset images are available, otherwise fall back to synthetic frames
            if is_defective and reject_images:
                img_path = random.choice(reject_images)
                filename = os.path.basename(img_path)
                with open(img_path, "rb") as f:
                    file_bytes = f.read()
            elif (not is_defective) and accept_images:
                img_path = random.choice(accept_images)
                filename = os.path.basename(img_path)
                with open(img_path, "rb") as f:
                    file_bytes = f.read()
            else:
                # Synthetic fallback if folders are empty
                if is_defective:
                    raw_frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
                else:
                    raw_frame = np.full((200, 200, 3), 180, dtype=np.uint8)
                    cv2.circle(raw_frame, (100, 100), 40, (140, 140, 140), -1)
                _, encoded_img = cv2.imencode(".jpg", raw_frame)
                file_bytes = encoded_img.tobytes()

            # Single post per unit to the inspection engine
            files = {"file": (filename, file_bytes, "image/jpeg")}
            requests.post(f"{API_BASE_URL}/api/inspect", files=files)

            # Query dynamic inventory calculations every 3 units or on the last unit
            if i % 3 == 0 or i == batch_size:
                inv_payload = {
                    "avg_daily_demand": 500.0,
                    "std_daily_demand": 60.0,
                    "avg_lead_time_days": 7.0,
                    "std_lead_time_days": 1.5
                }
                inv_res = requests.post(f"{API_BASE_URL}/api/inventory-status", json=inv_payload).json()
                
                prod = inv_res["production_metrics"]
                opt = inv_res["inventory_optimization"]

                metric_units.metric("Total Inspected", prod["total_inspected_units"])
                metric_scrap.metric("Scrap Rate", f"{prod['current_scrap_rate_pct']}%")
                metric_dyn_ss.metric("Dynamic SS", f"{opt['dynamic_safety_stock_units']} units")

                chart_data.append({
                    "Unit": i,
                    "Base Safety Stock": opt["base_safety_stock_units"],
                    "Dynamic Safety Stock": opt["dynamic_safety_stock_units"]
                })
                chart_placeholder.line_chart(pd.DataFrame(chart_data).set_index("Unit"))

                if opt["reorder_trigger_active"]:
                    alert_placeholder.error(
                        f"🚨 Dynamic Reorder Point ({opt['reorder_point_units']} units) breached current inventory ({opt['current_inventory']} units)!"
                    )

            prog_bar.progress(i / batch_size)
            status_text.text(f"Processing SKU Frame #{i} of {batch_size}...")
            time.sleep(stream_delay)

        status_text.text("✅ Conveyor Batch Inspection Complete.")

# =========================================================
# TAB 2: MANUAL SINGLE-FRAME INSPECTION (YOLO Powered)
# =========================================================
with tab_manual:
    st.subheader("Manual Vision Diagnostics (YOLOv8 Inference)")
    uploaded_file = st.file_uploader("Upload Frame", type=["jpg", "jpeg", "png"], key="manual_upload")
    
    if uploaded_file:
        col_in, col_out = st.columns(2)
        
        with col_in:
            st.image(uploaded_file, caption="Raw Ingested Frame", use_container_width=True)
            run_btn = st.button("Run YOLO Inspection", type="primary", use_container_width=True)

        if run_btn:
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            with st.spinner("Running YOLOv8 inference..."):
                res = requests.post(f"{API_BASE_URL}/api/inspect", files=files).json()
            
            with col_out:
                if "annotated_image_base64" in res and res["annotated_image_base64"]:
                    img_data = base64.b64decode(res["annotated_image_base64"])
                    st.image(img_data, caption="YOLOv8 Detection Overlay", use_container_width=True)
                
                if res.get("status") == "PASS":
                    st.success(f"**STATUS: PASS** | Detections: {res.get('detected_objects', [])}")
                else:
                    st.error(f"**STATUS: REJECT** | Defect: {res.get('defect_type', 'UNKNOWN')}")
            
            with st.expander("Full Telemetry Payload"):
                st.json({k: v for k, v in res.items() if k != "annotated_image_base64"})