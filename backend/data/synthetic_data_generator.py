import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class ClassroomDataGenerator:
    def __init__(self, num_students=30, num_classes=100):
        self.num_students = num_students
        self.num_classes = num_classes
        
    def generate_student_profiles(self):
        # each student gets a profile with gpa,prev scores and device type
        students = []
        subjects = ['Math', 'Science', 'English', 'History', 'CS']
        
        for i in range(self.num_students):
            student = {
                'student_id': f'STU_{i:03d}',
                'grade_level': random.choice([9, 10, 11, 12]),
                'gpa': np.random.normal(3.0, 0.5),
                'prev_scores': {
                    subject: max(0, min(100, np.random.normal(75, 15)))
                    for subject in subjects
                },
                'learning_pace': np.random.choice(['slow', 'medium', 'fast'], 
                                                  p=[0.2, 0.6, 0.2]),
                'device_type': np.random.choice(['laptop', 'tablet', 'both'], 
                                               p=[0.6, 0.3, 0.1])
            }
            students.append(student)
        return pd.DataFrame(students)
    
    def generate_class_session(self, class_id, subject, hour, student_profiles):
        # generates one full class session for all students
        session_data = []
        
        # post lunch and end of day students are naturally less engaged
        time_factor = 1.0
        if 12 <= hour <= 14:
            time_factor = 0.7
        elif hour >= 15:
            time_factor = 0.8
            
        subject_factors = {
            'Math': 0.9, 'Science': 0.95, 
            'English': 0.85, 'History': 0.8, 'CS': 1.0
        }
        
        for _, student in student_profiles.iterrows():
            # engagement is a weighted combo of gpa,prev score,time and subject
            base_engagement = (
                (student['gpa'] / 4.0) * 0.3 +
                (student['prev_scores'][subject] / 100) * 0.3 +
                time_factor * 0.2 +
                subject_factors[subject] * 0.2
            )
            
            # add noise so its not perfectly deterministic
            engagement = max(0, min(1, 
                base_engagement + np.random.normal(0, 0.1)))
            
            session_row = {
                'class_id': class_id,
                'subject': subject,
                'hour': hour,
                'student_id': student['student_id'],
                'engagement_score': engagement,
                # more disengaged = more app switches
                'app_switches': int(np.random.poisson(5 * (1 - engagement))),
                'keystroke_intensity': int(engagement * 100 + np.random.normal(0, 10)),
                'poll_participation': 1 if engagement > 0.3 else 0,
                'quiz_score': max(0, min(100, 
                    engagement * 100 + np.random.normal(0, 10))),
                'collaboration_actions': int(np.random.poisson(engagement * 10)),
                # more disengaged = more inactivity
                'inactivity_periods': int(np.random.poisson(3 * (1 - engagement))),
                'activity_type': np.random.choice(
                    ['lecture', 'group_work', 'individual_task', 'quiz'],
                    p=[0.3, 0.3, 0.3, 0.1]
                )
            }
            session_data.append(session_row)
            
        return pd.DataFrame(session_data)
    
    def generate_full_dataset(self):
        print("Generating student profiles...")
        students_df = self.generate_student_profiles()
        
        all_sessions = []
        subjects = ['Math', 'Science', 'English', 'History', 'CS']
        
        print("Generating class sessions...")
        for class_id in range(self.num_classes):
            subject = random.choice(subjects)
            hour = random.choice([8, 9, 10, 11, 12, 13, 14, 15])
            
            session_df = self.generate_class_session(
                f'CLASS_{class_id:03d}',
                subject,
                hour,
                students_df
            )
            all_sessions.append(session_df)
            
        final_df = pd.concat(all_sessions, ignore_index=True)
        
        # inject anomalies into 5% of rows to simulate sudden disengagement
        print("Injecting anomalies...")
        anomaly_indices = random.sample(range(len(final_df)), int(len(final_df) * 0.05))
        # cast to float first otherwise pandas 2.x throws dtype errors
        final_df['engagement_score'] = final_df['engagement_score'].astype(float)
        final_df['app_switches'] = final_df['app_switches'].astype(float)
        final_df['keystroke_intensity'] = final_df['keystroke_intensity'].astype(float)
        for idx in anomaly_indices:
            final_df.loc[idx, 'engagement_score'] *= 0.3
            final_df.loc[idx, 'app_switches'] *= 3
            final_df.loc[idx, 'keystroke_intensity'] *= 0.2
            
        return final_df, students_df

# Generate and save data
if __name__ == "__main__":
    generator = ClassroomDataGenerator(num_students=30, num_classes=200)
    sessions_df, students_df = generator.generate_full_dataset()
    
    sessions_df.to_csv('backend/data/synthetic_classroom_data.csv', index=False)
    students_df.to_csv('backend/data/student_profiles.csv', index=False)
    print("Synthetic data generated and saved!")