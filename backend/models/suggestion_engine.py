# backend/models/suggestion_engine.py
# ============================================================
# Context-aware suggestion engine.
# Given an activity type + severity, returns a specific,
# actionable suggestion the teacher can act on immediately.
# ============================================================

import random
from typing import Optional


class SuggestionEngine:
    """
    Rule-based suggestion engine (no ML needed here — rules are
    based on pedagogy research, not data).

    generate_suggestion() is the main method called by the API.
    """

    SUGGESTIONS = {
        "lecture": [
            "Ask {student} a direct question about the topic just covered.",
            "Request {student} to summarise the last key point in their own words.",
            "Use a quick show-of-hands poll and ask {student} to explain their choice.",
            "Call on {student} for a think-pair-share with their neighbour.",
            "Switch to a short visual aid or diagram to re-engage {student}.",
        ],
        "group_work": [
            "Assign {student} a specific role: timekeeper or note-taker.",
            "Ask {student}'s group to briefly share their progress with the class.",
            "Rotate group roles and make {student} the discussion leader.",
            "Give {student}'s group a 2-minute challenge question.",
            "Check privately whether {student} understands their assigned task.",
        ],
        "individual_task": [
            "Check if {student} needs clarification on the task requirements.",
            "Break the task into smaller numbered steps for {student}.",
            "Suggest {student} partners with a nearby student for this section.",
            "Offer {student} a hint or worked example to unblock their progress.",
            "Highlight just one key part of the task to reduce cognitive load for {student}.",
        ],
        "quiz": [
            "Encourage {student} — remind them to trust their instincts.",
            "Suggest {student} reviews their most recent notes for context.",
            "Remind {student} of the time remaining to help them focus.",
            "Offer {student} a brief strategy tip for answering difficult questions.",
        ],
    }

    def generate_suggestion(
        self,
        student_id: str,
        activity_type: str,
        anomaly_score: float,
        baseline_engagement: float,
        student_name: Optional[str] = None,
    ) -> dict:
        """
        Generate a context-aware suggestion.

        Args:
            student_id: The student's ID (used as fallback name).
            activity_type: lecture | group_work | individual_task | quiz
            anomaly_score: From the autoencoder (higher = more anomalous).
            baseline_engagement: Expected engagement from the RF model.
            student_name: Human-readable name if available.

        Returns:
            dict with keys: text, severity, type, anomaly_score, baseline_engagement
        """
        severity = "high" if anomaly_score > 2.0 else "medium"
        display_name = student_name or student_id

        # Fall back to lecture suggestions if activity_type unknown
        pool = self.SUGGESTIONS.get(activity_type, self.SUGGESTIONS["lecture"])
        text = random.choice(pool).format(student=display_name)

        return {
            "text": text,
            "severity": severity,
            "type": f"{activity_type}_low_engagement",
            "anomaly_score": round(float(anomaly_score), 3),
            "baseline_engagement": round(float(baseline_engagement), 3),
        }
