# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

with patch("backend.api.endpoints.load_models"):
    from backend.api.endpoints import app
    from backend.database import create_tables, engine, Base

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "ClassPulse API is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_start_session():
    response = client.post("/api/class/start", json={
        "class_id": "TEST_001",
        "teacher_id": "TCH_001",
        "subject": "Math"
    })
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_suggestion_engine():
    from backend.models.suggestion_engine import SuggestionEngine
    eng = SuggestionEngine()
    result = eng.generate_suggestion(
        student_id="STU_001",
        activity_type="lecture",
        anomaly_score=2.5,
        baseline_engagement=0.7,
        student_name="Ali"
    )
    assert "text" in result
    assert result["severity"] == "high"
    assert "Ali" in result["text"]


def test_engagement_calculation():
    from backend.api.endpoints import calculate_current_engagement, StudentActivity
    activity = StudentActivity(
        student_id="STU_001",
        app_switches=0,
        keystroke_intensity=100,
        poll_participation=1,
        collaboration_actions=30,
        inactivity_periods=0,
        activity_type="lecture",
        subject="Math",
        hour=10,
    )
    score = calculate_current_engagement(activity)
    assert 0.0 <= score <= 1.0