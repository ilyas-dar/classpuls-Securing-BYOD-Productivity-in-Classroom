# ClassPulse

**ML-powered classroom engagement monitoring for BYOD environments.**

ClassPulse uses machine learning to detect student disengagement in real time and gives teachers instant, context-aware suggestions — without monitoring content or violating privacy.

## Stack
- Backend: FastAPI + Uvicorn
- ML: scikit-learn (Random Forest + IsolationForest)
- Frontend: HTML / CSS / JS + Chart.js
- Deploy: Docker + Docker Compose

## Status
🚧 Active development — building incrementally.

## Structure (in progress)
\`\`\`
classpulse/
├── backend/        # FastAPI + ML models
├── frontend/       # Teacher dashboard
├── ml_pipeline/    # Training scripts
├── tests/          # Test suite
└── docker/         # Deployment
\`\`\`
