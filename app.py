import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Student Failure Prediction", layout="wide")

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('models/best_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    # explainer = joblib.load('models/explainer.pkl') # Loading explainer might be tricky if versions differ, safer to re-init if fast
    return model, scaler

try:
    model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please run `src/train_model.py` first.")
    st.stop()

def user_input_features():
    st.sidebar.header('Student Input Features')
    
    attendance = st.sidebar.slider('Attendance (%)', 0, 100, 75)
    internal_marks = st.sidebar.slider('Internal Marks (0-30)', 0, 30, 20)
    assignment_score = st.sidebar.slider('Assignment Score (0-10)', 0, 10, 8)
    lab_performance = st.sidebar.slider('Lab Performance (0-100)', 0, 100, 70)
    project_completion = st.sidebar.selectbox('Project Completion', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    semester_attempts = st.sidebar.selectbox('Semester Attempts', [1, 2, 3])
    
    study_hours = st.sidebar.slider('Study Hours/Week', 0, 20, 5)
    sleep_hours = st.sidebar.slider('Sleep Hours/Night', 0, 12, 7)
    stress_level = st.sidebar.slider('Stress Level (1-10)', 1, 10, 5)
    exam_anxiety = st.sidebar.slider('Exam Anxiety (1-10)', 1, 10, 5)
    skill_level = st.sidebar.slider('Skill Level (1-10)', 1, 10, 6)
    
    # Calculate CGPA roughly or ask user? The model expects it.
    # Let's ask user or infer. Model trained on it.
    # Let's add it to input
    cgpa = st.sidebar.slider('Current CGPA', 0.0, 10.0, 7.0)

    data = {
        'Attendance': attendance,
        'Internal_Marks': internal_marks,
        'CGPA': cgpa,
        'Assignment_Score': assignment_score,
        'Lab_Performance': lab_performance,
        'Study_Hours': study_hours,
        'Skill_Level': skill_level,
        'Stress_Level': stress_level,
        'Sleep_Hours': sleep_hours,
        'Exam_Anxiety': exam_anxiety,
        'Project_Completion': project_completion,
        'Semester_Attempts': semester_attempts
    }
    features = pd.DataFrame(data, index=[0])
    return features

st.title("🎓 Explainable Student Failure Prediction System")
st.markdown("""
This system predicts the likelihood of a student failing based on academic and behavioral metrics.
It uses **Explainable AI (XAI)** to provide insights into *why* a prediction was made.
""")

input_df = user_input_features()

st.subheader("Student Details")
st.write(input_df)

if st.button("Predict Result"):
    # Preprocess
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
    
    # Predict
    prediction = model.predict(input_scaled_df)
    probability = model.predict_proba(input_scaled_df)
    
    result = "FAIL" if prediction[0] == 1 else "PASS"
    prob_fail = probability[0][1]
    
    st.subheader("Prediction Result")
    if result == "FAIL":
        st.error(f"Prediction: **{result}** (Probability of Failure: {prob_fail:.2%})")
    else:
        st.success(f"Prediction: **{result}** (Probability of Failure: {prob_fail:.2%})")
        
    # Recommendations Section
    st.markdown("---")
    st.subheader("� Recommendations")
    
    # Rule-based recommendations based on input values
    risk_factors = []
    
    if input_df.iloc[0]['Attendance'] < 75:
        risk_factors.append(("Attendance", input_df.iloc[0]['Attendance'], "Consider improving attendance to at least 75%."))
    
    if input_df.iloc[0]['Internal_Marks'] < 15:
        risk_factors.append(("Internal Marks", input_df.iloc[0]['Internal_Marks'], "Focus on improving internal assessment scores to at least 15/30."))
    
    if input_df.iloc[0]['CGPA'] < 6.0:
        risk_factors.append(("CGPA", input_df.iloc[0]['CGPA'], "Work on improving overall CGPA to at least 6.0."))
    
    if input_df.iloc[0]['Study_Hours'] < 10:
        risk_factors.append(("Study Hours", input_df.iloc[0]['Study_Hours'], "Try to increase study hours to 10+ hours/week."))
    
    if input_df.iloc[0]['Stress_Level'] > 5:
        risk_factors.append(("Stress Level", input_df.iloc[0]['Stress_Level'], "High stress detected. Recommend counseling or relaxation techniques."))
    
    if input_df.iloc[0]['Exam_Anxiety'] > 5:
        risk_factors.append(("Exam Anxiety", input_df.iloc[0]['Exam_Anxiety'], "High exam anxiety. Consider stress management workshops."))
    
    if input_df.iloc[0]['Sleep_Hours'] < 6:
        risk_factors.append(("Sleep Hours", input_df.iloc[0]['Sleep_Hours'], "Insufficient sleep. Aim for at least 6-8 hours per night."))
    
    if input_df.iloc[0]['Assignment_Score'] < 6:
        risk_factors.append(("Assignment Score", input_df.iloc[0]['Assignment_Score'], "Improve assignment completion and quality."))
    
    if input_df.iloc[0]['Lab_Performance'] < 60:
        risk_factors.append(("Lab Performance", input_df.iloc[0]['Lab_Performance'], "Focus on improving lab work and practical skills."))
    
    if input_df.iloc[0]['Project_Completion'] == 0:
        risk_factors.append(("Project Completion", "No", "Complete your project on time - this is critical!"))
    
    if input_df.iloc[0]['Semester_Attempts'] > 1:
        risk_factors.append(("Semester Attempts", input_df.iloc[0]['Semester_Attempts'], "Multiple attempts detected. Seek academic support."))
    
    if risk_factors:
        st.warning(f"**{len(risk_factors)} Risk Factor(s) Identified:**")
        for feat, val, msg in risk_factors:
            st.write(f"- **{feat}** (Current: {val}): {msg}")
    else:
        st.success("✅ No major risk factors identified. Keep up the good work!")

