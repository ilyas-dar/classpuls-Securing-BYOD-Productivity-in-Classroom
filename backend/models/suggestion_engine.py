import random

class SuggestionEngine:
    def __init__(self):
        # suggestions grouped by activity type,keeps it context aware
        self.suggestion_bank = {
            'lecture': [
                "ask a direct question to re-engage the student",
                "use a quick think-pair-share activity",
                "launch a live poll so students vote anonymously",
                "request the student to summarize the last point",
                "switch to a short video clip or visual aid",
            ],
            'group_work': [
                "assign this student a specific role like timekeeper or note taker",
                "ask the group to share their progress out loud",
                "rotate group roles and make this student the discussion leader",
                "give the group a 2 minute challenge question",
                "check if the student understands their assigned task",
            ],
            'individual_task': [
                "check in privately,ask if they need clarification",
                "break the task into smaller numbered steps",
                "suggest pairing with a nearby student for this section",
                "offer a hint or worked example to unblock progress",
                "reduce cognitive load by highlighting just one key part",
            ],
            'quiz': [
                "encourage the student,remind them to trust their instincts",
                "suggest reviewing the most recent notes for context",
                "remind them of the time remaining to build focus",
                "offer a brief strategy tip for tackling difficult questions",
            ],
        }

    def get_suggestion(self, activity_type, engagement_score):
        # pick suggestion bank based on activity,fallback to lecture
        bank = self.suggestion_bank.get(activity_type, self.suggestion_bank['lecture'])
        text = random.choice(bank)
        # severity based on how low engagement actually is
        severity = 'high' if engagement_score < 0.5 else 'medium'
        return {
            'text': text,
            'severity': severity,
            'type': f'{activity_type}_low_engagement'
        }

    def generate_suggestion(self, student_id, activity_type, anomaly_score, baseline_engagement):
        # used by the api endpoint to generate suggestion for a flagged student
        severity = 'high' if anomaly_score > 2.0 else 'medium'
        bank = self.suggestion_bank.get(activity_type, self.suggestion_bank['lecture'])
        text = random.choice(bank)
        return {
            'text': text,
            'severity': severity,
            'type': f'{activity_type}_low_engagement',
            'student_id': student_id,
            'anomaly_score': round(anomaly_score, 3),
            'baseline_engagement': round(baseline_engagement, 3)
        }