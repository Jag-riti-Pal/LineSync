# 🏭 LineSync: FMCG Quality Assurance & Dynamic Replenishment Engine

> **A closed-loop industrial cyber-physical architecture connecting edge computer vision defect classification directly to dynamic safety stock recalculation and automated reorder points.**

---

## 📌 Executive Summary & Problem Context
In automated Fast-Moving Consumer Goods (FMCG) packaging lines (bottling, canning, and dry packaging), quality control at the factory floor is fundamentally disconnected from Sales & Operations Planning (S&OP) inventory logic.

When packaging machines produce defective units such as dented aluminum cans, crushed PET bottles, or conveyor line jams traditional Enterprise Resource Planning (ERP) databases assume that planned inventory matches actual output. Static safety stock equations do not account for immediate scrap volatility. As scrap rates increase, actual warehouse yield drops unexpectedly, triggering stockouts and supply-chain bottlenecks across regional distribution hubs.

**LineSync** bridges this gap. It captures real-time edge camera telemetry from high-speed packaging lines, classifies structural defects via a dual-metric Computer Vision pipeline (YOLOv8 + boundary irregularity/gradient analysis), and dynamically rescales inventory safety buffers using real-time scrap rates to trigger automated replenishment purchase orders.

---

## 🏗️ System Architecture

                     [ High-Speed Conveyor Stream ]
                                  │
                                  ▼
           [ Edge Computer Vision Inspector: vision_inspector.py ]
           ├── YOLOv8 SKU & Foreign Object Classifier (yolov8n.pt)
           └── Contour Jaggedness & Laplacian Variance Engine
                                  │
                                  │ (Classification, Confidence, HUD Frame)
                                  ▼
             [ Asynchronous Microservice API: main.py (FastAPI) ]
             ├── POST /api/inspect          (Ingest Frame & Log Telemetry)
             ├── POST /api/reset            (Flush Historical Memory)
             └── POST /api/inventory-status (Recalculate Replenishment)
                                  │
                                  ▼
            [ Dynamic Prescriptive Engine: inventory_engine.py ]
            ├── Rolling Scrap Rate Computation
            ├── Dynamic Safety Stock Inflation
            └── Reorder Point (ROP) Threshold Evaluation
                                  │
                                  ▼
          [ Interactive Operations Console: app.py (Streamlit) ]
          ├── Live Conveyor Run Simulation (Real Image Batches)
          ├── Real-time Dynamic SS vs Base SS Line Tracking
          └── Visual Single-Frame Diagnostic Inspection

---

## 📐 Mathematical Formulation

### 1. Classical Safety Stock Under Demand and Lead Time Uncertainty
Classical inventory safety stocks ($SS_{\text{base}}$) protect against deviations in daily consumer demand and supplier lead times:

$$SS_{\text{base}} = Z \times \sqrt{\bar{L} \cdot \sigma_D^2 + \bar{D}^2 \cdot \sigma_L^2}$$

* $Z$: Inverse cumulative standard normal distribution value for desired cycle-service level ($Z = 1.65$ represents a **95%** service level).
* $\bar{D}, \sigma_D$: Mean and standard deviation of daily SKU customer demand.
* $\bar{L}, \sigma_L$: Mean and standard deviation of supplier replenishment lead time (days).

### 2. Operational Line Scrap Calculation
From the continuous inspection telemetry, the empirical line scrap rate is calculated across the total inspected batch:

$$\text{Scrap Rate} = \frac{\sum \text{REJECT Units}}{\sum \text{Total Units Inspected}}$$

To maintain numerical stability in production and prevent division-by-zero errors when an entire batch is lost, the usable scrap rate is bound:

$$\text{Scrap}_{\text{eff}} = \min\left(\max(\text{Scrap Rate}, 0.0), 0.80\right)$$

### 3. Dynamic Safety Stock Rescaling
To protect downstream service levels when line yield degrades, safety stock scales inversely with line scrap:

$$SS_{\text{dynamic}} = \frac{SS_{\text{base}}}{1 - \text{Scrap}_{\text{eff}}}$$

### 4. Dynamic Reorder Point (ROP) & Purchase Order Alert
$$ROP = (\bar{D} \times \bar{L}) + SS_{\text{dynamic}}$$

$$\text{Trigger Reorder Alert} =  \begin{cases}  \text{TRUE}, & \text{if } \text{Inventory}_{\text{current}} \le ROP \\  \text{FALSE}, & \text{otherwise}  \end{cases}$$

---

## 🔬 Computer Vision Defect Detection Pipeline

To handle transparent PET plastic, printed reflective aluminum cans, and non-target anomalies, the inspection engine uses a calibrated multi-signal strategy:

1. **YOLOv8 Semantic Filtering:** Inspects incoming frames for valid FMCG packaging categories (`bottle`, `can`). If a foreign object or label anomaly is detected, the unit is immediately flagged as a `FOREIGN_OBJECT_ANOMALY`.
2. **Boundary Contour Jaggedness:** Computes the perimeter-to-area ratio of detected contours to identify surface crumpling and crushed aluminum:
   $$\text{Jaggedness} = \frac{P^2}{A + \epsilon}$$
   Crushed cans exhibit extreme boundary deformation ($\text{Jaggedness} \ge 450.0$).
3. **Laplacian Gradient Variance (Surface Crease Profiling):** Computes the variance of the Laplacian over the grayscale region of interest to detect texture degradation:
   $$\text{Wrinkle Var} = \text{Var}\left(\nabla^2 f(x, y)\right)$$
   Transparent PET bottles that collapse flatten their visual edges and drop into low contrast ($\text{Wrinkle Var} < 65.0$).

---

⚡ Quickstart Guide

Option A: Local Environment (Python 3.10+)

Clone the repository:
git clone [https://github.com/your-username/LineSync.git](https://github.com/your-username/LineSync.git)
cd LineSync

Set up a virtual environment and install dependencies:
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Start the FastAPI Microservice (Terminal 1):
uvicorn main:app --reload --host 127.0.0.1 --port 8000

Launch the Operations Console (Terminal 2):
streamlit run app.py

Access the interactive web console at http://localhost:8501.

Option B: Docker DeploymentDeploy both the backend microservice and frontend console using Docker Compose:
docker compose up --build

FastAPI Backend: 
http://localhost:8000 (API Docs: http://localhost:8000/docs)
Streamlit Console: http://localhost:8501

📊 Operations Console Features
Conveyor Batch Simulator: Streams batches of real packaging frames from data/good/ and data/bad/ at user-defined line speeds and injected defect probabilities.
Live Telemetry & Closed-Loop Charts: Plots real-time adjustments comparing SS_base against SS_dynamic as rejections occur.
ERP Alert Triggering: Flags red alert warnings the moment the inflated Reorder Point exceeds available stock.
Single-Frame Manual Diagnostics: Allows plant operators to upload individual images to view YOLO bounding boxes, defect tags, and confidence scores.

---

## 📂 Repository Structure

```text
LineSync/
├── main.py                 # FastAPI microservice routing inspection & inventory logic
├── app.py                  # Dual-panel Streamlit operations dashboard & simulator
├── vision_inspector.py     # YOLOv8 + OpenCV contour/gradient anomaly engine
├── inventory_engine.py     # Deterministic safety stock & ROP replenishment calculations
├── caliberate.py            # Dataset diagnostic profile feature extractor
├── Dockerfile              # Containerization definition with OpenCV C-libraries
├── docker_compose.yml      # Orchestration for FastAPI + Streamlit deployment
├── requirements.txt        # Pinned runtime dependencies
├── README.md               # Architecture documentation and mathematical proofs
└── data/
    ├── good/               # Intact sample bottles and cans
    └── bad/                # Defective (crushed, dented, collapsed) samples





