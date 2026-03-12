# backend/data/synthetic_data_generator.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class ClassroomDataGenerator:
    def __init__(self, num_students=30, num_classes=100):
        self.num_students = num_students
        self.num_classes = num_classes
        
    def generate_student_profiles(self):
        """Generate static student data"""
        students = []
        subjects = ['Math', 'Science', 'English', 'History', 'CS']
        
        for i in range(self.num_students):
            # Create base student profile
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
        """Generate data for one class session"""
        session_data = []
        
        # Base engagement factors
        time_factor = 1.0
        if 12 <= hour <= 14:  # Post-lunch dip
            time_factor = 0.7
        elif hour >= 15:  # End of day fatigue
            time_factor = 0.8
            
        subject_factors = {
            'Math': 0.9, 'Science': 0.95, 
            'English': 0.85, 'History': 0.8, 'CS': 1.0
        }
        
        for _, student in student_profiles.iterrows():
            # Calculate base engagement
            base_engagement = (
                (student['gpa'] / 4.0) * 0.3 +
                (student['prev_scores'][subject] / 100) * 0.3 +
                time_factor * 0.2 +
                subject_factors[subject] * 0.2
            )
            
            # Add some randomness
            engagement = max(0, min(1, 
                base_engagement + np.random.normal(0, 0.1)))
            
            # Generate BYOD metrics based on engagement
            session_row = {
                'class_id': class_id,
                'subject': subject,
                'hour': hour,
                'student_id': student['student_id'],
                'engagement_score': engagement,
                'app_switches': int(np.random.poisson(5 * (1 - engagement))),
                'keystroke_intensity': int(engagement * 100 + np.random.normal(0, 10)),
                'poll_participation': 1 if engagement > 0.3 else 0,
                'quiz_score': max(0, min(100, 
                    engagement * 100 + np.random.normal(0, 10))),
                'collaboration_actions': int(np.random.poisson(engagement * 10)),
                'inactivity_periods': int(np.random.poisson(3 * (1 - engagement))),
                'activity_type': np.random.choice(
                    ['lecture', 'group_work', 'individual_task', 'quiz'],
                    p=[0.3, 0.3, 0.3, 0.1]
                )
            }
            session_data.append(session_row)
            
        return pd.DataFrame(session_data)
    
    def generate_full_dataset(self):
        """Generate complete training dataset"""
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
        
        # Add some anomalies (students who suddenly disengage)
        print("Injecting anomalies...")
        anomaly_indices = random.sample(range(len(final_df)), int(len(final_df) * 0.05))
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
    
    # Save to CSV
    sessions_df.to_csv('backend/data/synthetic_classroom_data.csv', index=False)
    students_df.to_csv('backend/data/student_profiles.csv', index=False)
    print("Synthetic data generated and saved!")
