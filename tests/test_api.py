# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Mock DB and models before importing app
with patch("backend.database.create_tables"), \
     patch("backend.api.endpoints.load_models"):
    from backend.api.endpoints import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "ClassPulse API is running"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_start_session():
    with patch("backend.api.endpoints.get_db") as mock_db:
        mock_db.return_value.__enter__ = lambda s: s
        mock_db.return_value.__exit__ = lambda s, *a: None
        response = client.post("/api/class/start", json={
            "class_id": "TEST_001",
            "teacher_id": "TCH_001",
            "subject": "Math"
        })
        assert response.status_code == 200

def test_suggestion_engine():
    from backend.models.suggestion_engine import SuggestionEngine
    engine = SuggestionEngine()
    result = engine.generate_suggestion(
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
    from datetime import datetime
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