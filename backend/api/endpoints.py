# backend/api/endpoints.py
# ============================================================
# Main FastAPI application.
#
# What changed from the original simulation:
#   - Models are loaded safely (warns if not trained yet, never crashes)
#   - All activity data is saved to PostgreSQL (not lost on restart)
#   - WebSocket uses the shared ConnectionManager
#   - Imports use absolute paths (works in Docker, tests, locally)
#   - Config comes from environment variables via backend/config.py
# ============================================================

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database import get_db, create_tables, ActivityRecord, Alert, ClassSession as DBSession
from backend.models.suggestion_engine import SuggestionEngine
from backend.api.websocket import manager

logger = logging.getLogger(__name__)
settings = get_settings()

# ============================================================
# App setup
# ============================================================

app = FastAPI(
    title="ClassPulse API",
    description="ML-powered classroom engagement monitoring",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Model loading (lazy — warns if not trained yet)
# ============================================================

baseline_model = None
anomaly_detector = None
suggestion_engine = SuggestionEngine()


def load_models():
    """Load trained models from disk. Called once at startup."""
    global baseline_model, anomaly_detector

    baseline_path = os.path.join(settings.model_path, "baseline_model.pkl")
    anomaly_prefix = os.path.join(settings.model_path, "anomaly_detector")

    if os.path.exists(baseline_path):
        from backend.models.baseline_model import BaselineEngagementModel
        baseline_model = BaselineEngagementModel.load(baseline_path)
        logger.info("Baseline model loaded ✓")
    else:
        logger.warning(
            "Baseline model not found at %s — run ml_pipeline/training/train_baseline.py first",
            baseline_path,
        )

    if os.path.exists(f"{anomaly_prefix}_model.keras") or os.path.exists(f"{anomaly_prefix}_model.h5"):
        from backend.models.anomaly_detector import AnomalyDetector
        anomaly_detector = AnomalyDetector.load(anomaly_prefix)
        logger.info("Anomaly detector loaded ✓")
    else:
        logger.warning(
            "Anomaly detector not found at %s_model.h5 — run ml_pipeline/training/train_anomaly.py first",
            anomaly_prefix,
        )


@app.on_event("startup")
def startup_event():
    create_tables()   # Create DB tables if they don't exist
    load_models()     # Load ML models if trained


# ============================================================
# Pydantic request / response schemas
# ============================================================

class StudentActivity(BaseModel):
    student_id: str
    app_switches: int
    keystroke_intensity: int
    poll_participation: int
    quiz_score: Optional[float] = None
    collaboration_actions: int
    inactivity_periods: int
    activity_type: str       # lecture | group_work | individual_task | quiz
    subject: str
    hour: int
    student_name: Optional[str] = None


class StartSessionRequest(BaseModel):
    class_id: str
    teacher_id: str
    subject: str


class ActivityResponse(BaseModel):
    status: str
    baseline_engagement: float
    current_engagement: float
    anomaly_score: float
    alert_generated: bool


# ============================================================
# Helper functions
# ============================================================

def calculate_current_engagement(activity: StudentActivity) -> float:
    """
    Simple heuristic to estimate current engagement from raw metrics.
    This runs even when ML models are not loaded.
    """
    score = (
        (1 - min(activity.app_switches, 20) / 20) * 0.3
        + (min(activity.keystroke_intensity, 100) / 100) * 0.3
        + activity.poll_participation * 0.2
        + (min(activity.collaboration_actions, 30) / 30) * 0.2
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def get_baseline_engagement(activity: StudentActivity) -> float:
    """Call the RF model, or return a default if not loaded."""
    if baseline_model is None:
        return 0.7  # sensible default

    import pandas as pd
    # Build the same feature vector the training script uses
    row = pd.DataFrame([{
        "hour": activity.hour,
        "app_switches": activity.app_switches,
        "keystroke_intensity": activity.keystroke_intensity,
        "poll_participation": activity.poll_participation,
        "collaboration_actions": activity.collaboration_actions,
        "inactivity_periods": activity.inactivity_periods,
    }])
    try:
        return float(baseline_model.predict(row)[0])
    except Exception as e:
        logger.warning("Baseline prediction failed: %s", e)
        return 0.7


def get_anomaly_score(activity: StudentActivity) -> float:
    """Call the autoencoder, or return 0 if not loaded."""
    if anomaly_detector is None:
        return 0.0
    try:
        return float(anomaly_detector.score(activity.dict()))
    except Exception as e:
        logger.warning("Anomaly scoring failed: %s", e)
        return 0.0


# ============================================================
# Routes
# ============================================================

@app.get("/")
def root():
    return {
        "message": "ClassPulse API is running",
        "version": "1.0.0",
        "models_loaded": {
            "baseline": baseline_model is not None,
            "anomaly_detector": anomaly_detector is not None,
        },
    }


@app.get("/health")
def health():
    """Used by Docker health-check and AWS load balancer."""
    return {"status": "healthy"}


@app.post("/api/class/start")
def start_session(req: StartSessionRequest, db: Session = Depends(get_db)):
    """Teacher starts a new class session."""
    session = DBSession(
        id=req.class_id,
        teacher_id=req.teacher_id,
        subject=req.subject,
        is_active=True,
    )
    # Upsert — if the teacher restarts the same class, just reactivate it
    existing = db.query(DBSession).filter(DBSession.id == req.class_id).first()
    if existing:
        existing.is_active = True
        existing.started_at = datetime.utcnow()
    else:
        db.add(session)
    db.commit()
    return {"status": "success", "class_id": req.class_id}


@app.post("/api/student/activity", response_model=ActivityResponse)
async def process_activity(
    activity: StudentActivity,
    class_id: str,
    db: Session = Depends(get_db),
):
    """
    Main endpoint called by the BYOD tracker every 30 seconds.
    1. Compute ML scores
    2. Save to database
    3. If alert threshold crossed → save alert + push to teacher dashboard via WebSocket
    """
    baseline_eng = get_baseline_engagement(activity)
    anomaly_score = get_anomaly_score(activity)
    current_eng = calculate_current_engagement(activity)

    # Save activity to DB
    record = ActivityRecord(
        session_id=class_id,
        student_id=activity.student_id,
        app_switches=activity.app_switches,
        keystroke_intensity=activity.keystroke_intensity,
        poll_participation=activity.poll_participation,
        quiz_score=activity.quiz_score,
        collaboration_actions=activity.collaboration_actions,
        inactivity_periods=activity.inactivity_periods,
        activity_type=activity.activity_type,
        subject=activity.subject,
        hour=activity.hour,
        baseline_engagement=baseline_eng,
        current_engagement=current_eng,
        anomaly_score=anomaly_score,
    )
    db.add(record)

    alert_generated = False
    # Alert condition: anomalous behaviour AND significantly below baseline
    if anomaly_score > 1.5 and current_eng < baseline_eng * 0.7:
        suggestion = suggestion_engine.generate_suggestion(
            student_id=activity.student_id,
            activity_type=activity.activity_type,
            anomaly_score=anomaly_score,
            baseline_engagement=baseline_eng,
            student_name=activity.student_name,
        )

        alert = Alert(
            session_id=class_id,
            student_id=activity.student_id,
            suggestion_text=suggestion["text"],
            suggestion_type=suggestion["type"],
            severity=suggestion["severity"],
            anomaly_score=anomaly_score,
        )
        db.add(alert)
        alert_generated = True

        # Push to teacher dashboard in real time
        await manager.broadcast(class_id, {
            "type": "alert",
            "student_id": activity.student_id,
            "student_name": activity.student_name or activity.student_id,
            "suggestion": suggestion,
            "current_engagement": current_eng,
            "timestamp": datetime.utcnow().isoformat(),
        })

    db.commit()

    return ActivityResponse(
        status="processed",
        baseline_engagement=round(baseline_eng, 4),
        current_engagement=round(current_eng, 4),
        anomaly_score=round(anomaly_score, 4),
        alert_generated=alert_generated,
    )


@app.websocket("/ws/teacher/{class_id}")
async def teacher_ws(websocket: WebSocket, class_id: str):
    """Teacher dashboard connects here to receive live alerts."""
    await manager.connect(websocket, class_id)
    try:
        while True:
            # Keep connection alive; handle ping/dismiss messages from dashboard
            data = await websocket.receive_text()
            # Future: handle "dismiss alert" commands from teacher
    except WebSocketDisconnect:
        manager.disconnect(websocket, class_id)


@app.get("/api/class/{class_id}/alerts")
def get_alerts(class_id: str, db: Session = Depends(get_db)):
    """Get all undismissed alerts for a class session."""
    alerts = (
        db.query(Alert)
        .filter(Alert.session_id == class_id, Alert.dismissed == False)
        .order_by(Alert.timestamp.desc())
        .all()
    )
    return {
        "class_id": class_id,
        "alerts": [
            {
                "id": a.id,
                "student_id": a.student_id,
                "suggestion_text": a.suggestion_text,
                "severity": a.severity,
                "anomaly_score": a.anomaly_score,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in alerts
        ],
    }


@app.get("/api/class/{class_id}/summary")
def get_summary(class_id: str, db: Session = Depends(get_db)):
    """End-of-class summary: total alerts, unique students flagged, avg engagement."""
    alerts = db.query(Alert).filter(Alert.session_id == class_id).all()
    records = db.query(ActivityRecord).filter(ActivityRecord.session_id == class_id).all()

    avg_engagement = (
        round(float(np.mean([r.current_engagement for r in records if r.current_engagement])), 4)
        if records else 0.0
    )

    return {
        "class_id": class_id,
        "total_alerts": len(alerts),
        "unique_students_flagged": len({a.student_id for a in alerts}),
        "average_engagement": avg_engagement,
        "total_activity_records": len(records),
    }


@app.delete("/api/alerts/{alert_id}/dismiss")
def dismiss_alert(alert_id: int, db: Session = Depends(get_db)):
    """Teacher dismisses an alert from their dashboard."""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.dismissed = True
    db.commit()
    return {"status": "dismissed"}


# ── Teacher Feedback (training signal) ────────────────────────────────

class FeedbackRequest(BaseModel):
    alert_id: int
    session_id: str
    student_id: str
    action: str          # "helpful" | "not_helpful" | "acted_on"
    note: Optional[str] = None
    actual_engagement_improved: Optional[bool] = None


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """
    Teacher submits feedback on an alert suggestion.
    This data is stored and will be used to retrain the models.
    """
    from backend.database import TeacherFeedback
    feedback = TeacherFeedback(
        alert_id=req.alert_id,
        session_id=req.session_id,
        student_id=req.student_id,
        action=req.action,
        note=req.note,
        actual_engagement_improved=req.actual_engagement_improved,
    )
    db.add(feedback)
    db.commit()
    return {"status": "feedback recorded", "action": req.action}


@app.get("/api/feedback/stats")
def feedback_stats(db: Session = Depends(get_db)):
    """How helpful are our suggestions? Shows model performance in production."""
    from backend.database import TeacherFeedback
    from sqlalchemy import func
    results = db.query(
        TeacherFeedback.action,
        func.count(TeacherFeedback.id).label("count")
    ).group_by(TeacherFeedback.action).all()

    total = sum(r.count for r in results)
    stats = {r.action: r.count for r in results}
    helpful = stats.get("helpful", 0) + stats.get("acted_on", 0)
    return {
        "total_feedback": total,
        "breakdown": stats,
        "helpfulness_rate": round(helpful / total, 3) if total > 0 else None,
    }
