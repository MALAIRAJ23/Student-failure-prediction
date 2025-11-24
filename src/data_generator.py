import pandas as pd
import numpy as np
import random
import os

def generate_student_data(num_students=1000):
    np.random.seed(42)
    random.seed(42)
    
    os.makedirs('data', exist_ok=True)
    
    data = []
    
    for _ in range(num_students):
        # Academic Features
        attendance = np.random.randint(40, 100)
        internal_marks = np.random.randint(0, 30) # Max 30
        assignment_score = np.random.randint(0, 10) # Max 10
        lab_performance = np.random.randint(0, 100)
        project_completion = np.random.choice([0, 1], p=[0.1, 0.9]) # 0: No, 1: Yes
        semester_attempts = np.random.choice([1, 2, 3], p=[0.9, 0.08, 0.02])
        
        # Behavioral/Personal Features
        study_hours = np.random.randint(1, 15) # Hours per week
        sleep_hours = np.random.randint(4, 10) # Hours per night
        stress_level = np.random.randint(1, 11) # 1-10 scale
        exam_anxiety = np.random.randint(1, 11) # 1-10 scale
        skill_level = np.random.randint(1, 11) # 1-10 scale
        
        # Derived Logic for Target Variable (Pass/Fail)
        # We create a "score" to determine pass probability
        # Positive factors
        score = (attendance * 0.3) + (internal_marks * 1.5) + (assignment_score * 2) + \
                (lab_performance * 0.2) + (study_hours * 1.5) + (skill_level * 2)
        
        # Negative factors
        score -= (stress_level * 1.5) + (exam_anxiety * 1.5) + ((3 - semester_attempts) * -5)
        
        if project_completion == 0:
            score -= 20
            
        # Add some noise
        score += np.random.normal(0, 10)
        
        # Determine CGPA based on score (approximate mapping)
        cgpa = min(10, max(0, score / 10))
        
        # Determine Pass/Fail (1: Pass, 0: Fail)
        # Let's say threshold is around 50 score points for passing
        # But we want "Fail" to be the target class usually in detection, 
        # however user asked for "Pass/Fail label". Let's use 0 for Fail, 1 for Pass.
        # User asked "Predict Why a Student Fails", so maybe Target=1 means Fail?
        # Standard convention: 1 is the positive class (the thing we are detecting).
        # Let's set Target: 1 = Fail, 0 = Pass.
        
        # If score is low, they fail.
        threshold = 65 # Adjusted to get a balanced-ish dataset
        if score < threshold:
            target = 1 # Fail
        else:
            target = 0 # Pass
            
        data.append([
            attendance, internal_marks, cgpa, assignment_score, lab_performance,
            study_hours, skill_level, stress_level, sleep_hours, exam_anxiety,
            project_completion, semester_attempts, target
        ])
        
    columns = [
        'Attendance', 'Internal_Marks', 'CGPA', 'Assignment_Score', 'Lab_Performance',
        'Study_Hours', 'Skill_Level', 'Stress_Level', 'Sleep_Hours', 'Exam_Anxiety',
        'Project_Completion', 'Semester_Attempts', 'Target_Fail'
    ]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    output_path = 'd:/PROJECTS/ml-project/data/student_performance.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {num_students} records at {output_path}")
    print(df['Target_Fail'].value_counts())
    return df

if __name__ == "__main__":
    generate_student_data()
