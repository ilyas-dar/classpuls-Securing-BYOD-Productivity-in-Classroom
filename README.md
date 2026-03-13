# ClassPulse — Securing BYOD Productivity in Classroom

A machine learning system that monitors student engagement in real time during BYOD classroom sessions. Teachers get a live dashboard with alerts and suggestions when students disengage.

## What it does

- tracks engagement levels of 30 students across a class session
- detects sudden disengagement using anomaly detection
- gives teachers context aware suggestions based on current activity
- generates a full session report at the end with per student breakdown

## How it works

The system has two parts that work together:

**ML Pipeline** trains on synthetic classroom data to learn what engaged vs disengaged behavior looks like based on BYOD signals like app switches, keystroke intensity, inactivity periods and collaboration actions. We compared Random Forest and XGBoost and automatically pick the better model.

**Teacher Dashboard** shows live engagement scores, alerts and suggestions during a class session.

## Note on Simulation

The dashboard currently runs on a JavaScript simulation. This is because real BYOD data requires actual student devices running a tracker that sends signals to the backend. The ML models are fully trained and saved, and the backend WebSocket infrastructure is ready — the simulation is a realistic demo of what the live system would look like with real student devices connected.

In a real deployment, each student device would run a lightweight tracker that sends behavioral signals every few seconds to the backend, which runs them through the trained model and pushes results to the teacher dashboard.

## Stack

- Backend: FastAPI + Uvicorn
- ML: scikit-learn (Random Forest), XGBoost, IsolationForest
- Frontend: HTML + CSS + JavaScript + Chart.js
- Data: synthetic classroom data generated with numpy and pandas

## How to run

**Step 1 — generate data and train models**
```bash
cd classpulse_refined
source venv/bin/activate
export PYTHONPATH=$(pwd)

python backend/data/synthetic_data_generator.py
python ml_pipeline/training/train_baseline.py
python ml_pipeline/training/train_anomaly.py
```

**Step 2 — start backend**
```bash
uvicorn backend.api.endpoints:app --reload --port 8000 --app-dir backend
```

**Step 3 — start frontend**
```bash
cd frontend/teacher_dashboard
python -m http.server 8080
```

Open `http://localhost:8080` in your browser.

The frontend works without the backend too — it falls back to simulation mode automatically if the WebSocket connection fails.

## ML Models

| Model | MSE | RMSE | MAE | R2 |
|---|---|---|---|---|
| Random Forest | 0.0222 | 0.1488 | 0.1076 | 0.0691 |
| XGBoost | 0.0221 | 0.1486 | 0.1066 | 0.0727 |

XGBoost was selected as the final model. R2 is low because synthetic data has high randomness by design — in a real deployment with actual BYOD signals the model would perform significantly better.

## Project Structure
```
classpulse/
├── backend/
│   ├── api/          — FastAPI endpoints and WebSocket
│   ├── data/         — data generation and preprocessing
│   └── models/       — anomaly detector, suggestion engine
├── frontend/
│   ├── teacher_dashboard/   — live engagement dashboard
│   └── student_client/      — BYOD tracker (ready for real deployment)
├── ml_pipeline/
│   ├── training/     — train_baseline.py, train_anomaly.py
│   └── models/       — saved trained models
└── tests/
```

## Future Work

- connect real student devices via the BYOD tracker
- add historical session comparison
- per student engagement trend analysis over multiple classes
- mobile friendly teacher dashboard