# backend/data/synthetic_data_generator.py
# ============================================================
# Realistic synthetic classroom data generator.
#
# Key improvements over v1:
#   - Student archetypes (bright-but-bored, struggling-but-trying, etc.)
#   - Monday morning effect, post-lunch slump, end-of-week fatigue
#   - Subject difficulty affects engagement differently per archetype
#   - Session-level events (fire drill, surprise quiz, group energy)
#   - Realistic noise and within-student consistency over time
#
# Usage (from project root):
#   python -m backend.data.synthetic_data_generator
# ============================================================

import numpy as np
import pandas as pd
import random
import os
from dataclasses import dataclass

ARCHETYPES = {
    "high_achiever":          {"weight": 0.20, "base_engagement": 0.82, "gpa_range": (3.5, 4.0), "boredom_risk": 0.15, "stress_sensitivity": 0.3},
    "bright_but_bored":       {"weight": 0.10, "base_engagement": 0.55, "gpa_range": (3.2, 4.0), "boredom_risk": 0.60, "stress_sensitivity": 0.2},
    "steady_worker":          {"weight": 0.30, "base_engagement": 0.70, "gpa_range": (2.8, 3.5), "boredom_risk": 0.10, "stress_sensitivity": 0.4},
    "struggling_but_trying":  {"weight": 0.15, "base_engagement": 0.65, "gpa_range": (2.0, 2.8), "boredom_risk": 0.05, "stress_sensitivity": 0.7},
    "disengaged":             {"weight": 0.15, "base_engagement": 0.35, "gpa_range": (1.5, 2.5), "boredom_risk": 0.70, "stress_sensitivity": 0.1},
    "anxious_high_performer": {"weight": 0.10, "base_engagement": 0.75, "gpa_range": (3.3, 4.0), "boredom_risk": 0.05, "stress_sensitivity": 0.9},
}

SUBJECTS = ["Math", "Science", "English", "History", "CS"]
SUBJECT_DIFFICULTY = {"Math": 0.8, "Science": 0.7, "CS": 0.75, "English": 0.5, "History": 0.45}
ACTIVITY_TYPES = ["lecture", "group_work", "individual_task", "quiz"]

ACTIVITY_MODS = {
    "lecture":         {"bright_but_bored": -0.20, "disengaged": -0.15, "anxious_high_performer": +0.05, "high_achiever": +0.05},
    "group_work":      {"struggling_but_trying": +0.10, "disengaged": +0.05, "bright_but_bored": +0.10},
    "individual_task": {"anxious_high_performer": -0.10, "struggling_but_trying": -0.15, "high_achiever": +0.05},
    "quiz":            {"anxious_high_performer": -0.25, "struggling_but_trying": -0.20, "disengaged": -0.10, "high_achiever": +0.05},
}


def time_factor(hour: int, day_of_week: int) -> float:
    h = {8: 0.85, 9: 0.95, 10: 1.05, 11: 1.05, 12: 0.70, 13: 0.75, 14: 0.90, 15: 0.75}.get(hour, 0.85)
    d = {0: 0.85, 1: 1.0, 2: 1.0, 3: 0.95, 4: 0.80}.get(day_of_week, 1.0)
    return float(np.clip(h * d, 0.5, 1.1))


@dataclass
class StudentProfile:
    student_id: str
    archetype: str
    gpa: float
    grade_level: int
    learning_pace: str
    prev_scores: dict
    base_engagement: float
    boredom_risk: float
    stress_sensitivity: float


class ClassroomDataGenerator:
    def __init__(self, num_students: int = 40, num_sessions: int = 300):
        self.num_students = num_students
        self.num_sessions = num_sessions

    def generate_students(self) -> list:
        students = []
        archetype_names = list(ARCHETYPES.keys())
        weights = [ARCHETYPES[a]["weight"] for a in archetype_names]
        for i in range(self.num_students):
            aname = random.choices(archetype_names, weights=weights)[0]
            a = ARCHETYPES[aname]
            gpa = float(np.clip(np.random.uniform(*a["gpa_range"]) + np.random.normal(0, 0.1), 0.0, 4.0))
            prev_scores = {}
            for subj in SUBJECTS:
                diff = SUBJECT_DIFFICULTY[subj]
                base = gpa / 4.0 * 100
                penalty = diff * (1 - gpa / 4.0) * 20
                prev_scores[subj] = float(np.clip(base - penalty + np.random.normal(0, 8), 20, 100))
            students.append(StudentProfile(
                student_id=f"STU_{i:03d}", archetype=aname, gpa=gpa,
                grade_level=random.choice([9, 10, 11, 12]),
                learning_pace=random.choices(["slow", "medium", "fast"], weights=[0.25, 0.55, 0.20])[0],
                prev_scores=prev_scores, base_engagement=a["base_engagement"],
                boredom_risk=a["boredom_risk"], stress_sensitivity=a["stress_sensitivity"],
            ))
        return students

    def compute_engagement(self, student, subject, hour, day_of_week, activity_type, session_energy) -> float:
        tf = time_factor(hour, day_of_week)
        subj_factor = 0.7 + 0.3 * (student.prev_scores[subject] / 100)
        activity_mod = ACTIVITY_MODS.get(activity_type, {}).get(student.archetype, 0.0)
        boredom_penalty = -0.25 if np.random.random() < student.boredom_risk else 0.0
        engagement = (student.base_engagement * tf * subj_factor + activity_mod + boredom_penalty
                      + session_energy * 0.1 + np.random.normal(0, 0.05))
        return float(np.clip(engagement, 0.0, 1.0))

    def engagement_to_metrics(self, engagement, student, activity_type) -> dict:
        base_switches = max(0, np.random.poisson(6 * (1 - engagement)))
        if student.archetype == "bright_but_bored":
            base_switches = int(base_switches * 1.5)
        app_switches = int(np.clip(base_switches, 0, 30))

        keystroke_base = engagement * 100
        if activity_type == "individual_task": keystroke_base *= 1.3
        elif activity_type == "lecture": keystroke_base *= 0.7
        keystroke_intensity = int(np.clip(keystroke_base + np.random.normal(0, 12), 0, 200))

        poll_threshold = 0.4 if student.archetype != "disengaged" else 0.6
        poll_participation = 1 if engagement > poll_threshold else 0

        quiz_score = float(np.clip(engagement * 70 + student.gpa / 4.0 * 30 + np.random.normal(0, 8), 0, 100))

        collab_base = engagement * 15 if activity_type == "group_work" else engagement * 3
        collaboration_actions = int(np.clip(np.random.poisson(collab_base), 0, 40))

        inactivity_base = (1 - engagement) * 4
        if student.archetype == "struggling_but_trying" and engagement < 0.5:
            inactivity_base *= 2
        inactivity_periods = int(np.clip(np.random.poisson(inactivity_base), 0, 15))

        return {"app_switches": app_switches, "keystroke_intensity": keystroke_intensity,
                "poll_participation": poll_participation, "quiz_score": quiz_score,
                "collaboration_actions": collaboration_actions, "inactivity_periods": inactivity_periods}

    def generate(self):
        print(f"Generating {self.num_students} student profiles with archetypes...")
        students = self.generate_students()
        from collections import Counter
        for arch, cnt in Counter(s.archetype for s in students).items():
            print(f"  {arch}: {cnt}")

        print(f"\nGenerating {self.num_sessions} class sessions...")
        rows = []
        for i in range(self.num_sessions):
            subject = random.choice(SUBJECTS)
            hour = random.choice([8, 9, 10, 11, 12, 13, 14, 15])
            day_of_week = random.randint(0, 4)
            activity_type = random.choices(ACTIVITY_TYPES, weights=[0.35, 0.25, 0.30, 0.10])[0]
            class_id = f"CLASS_{i:04d}"
            session_energy = np.random.normal(0, 0.15)
            for student in students:
                eng = self.compute_engagement(student, subject, hour, day_of_week, activity_type, session_energy)
                metrics = self.engagement_to_metrics(eng, student, activity_type)
                rows.append({"class_id": class_id, "student_id": student.student_id,
                             "archetype": student.archetype, "subject": subject, "hour": hour,
                             "day_of_week": day_of_week, "activity_type": activity_type,
                             "grade_level": student.grade_level, "gpa": student.gpa,
                             "learning_pace": student.learning_pace,
                             "prev_score": student.prev_scores[subject],
                             "engagement_score": eng, **metrics})

        df = pd.DataFrame(rows)
        profiles_df = pd.DataFrame([{
            "student_id": s.student_id, "archetype": s.archetype, "gpa": s.gpa,
            "grade_level": s.grade_level, "learning_pace": s.learning_pace,
            **{f"prev_{subj}": s.prev_scores[subj] for subj in SUBJECTS},
        } for s in students])

        print(f"\nDataset summary:")
        print(f"  Total records : {len(df)}")
        print(f"  Avg engagement: {df['engagement_score'].mean():.3f}")
        print(f"  Std engagement: {df['engagement_score'].std():.3f}")
        print(f"  Anomaly rate  : {(df['engagement_score'] < 0.4).mean():.1%}")
        return df, profiles_df


if __name__ == "__main__":
    gen = ClassroomDataGenerator(num_students=40, num_sessions=300)
    sessions_df, profiles_df = gen.generate()
    out_dir = os.path.dirname(__file__)
    sessions_df.to_csv(os.path.join(out_dir, "synthetic_classroom_data.csv"), index=False)
    profiles_df.to_csv(os.path.join(out_dir, "student_profiles.csv"), index=False)
    print("Saved → backend/data/synthetic_classroom_data.csv")
    print("Saved → backend/data/student_profiles.csv")
