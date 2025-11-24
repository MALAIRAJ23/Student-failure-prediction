# Presentation Content: Explainable Student Failure Prediction

## Slide 1: Title Slide
- **Title**: Explainable ML System to Predict Why a Student Fails
- **Subtitle**: XAI-Failure Prediction Model
- **Presenter**: [Your Name]
- **Date**: [Current Date]

## Slide 2: Problem Statement
- **The Issue**: Student failure rates in higher education are a major concern.
- **Current Gap**: Existing systems predict *who* might fail but rarely explain *why* (Black Box problem).
- **Consequence**: Interventions are generic and often ineffective.
- **Solution**: An XAI-powered system that provides interpretable reasons for failure risk.

## Slide 3: Objectives
1. **Predict**: Accurately identify at-risk students using ML.
2. **Explain**: Use SHAP/LIME to reveal the "why" behind predictions.
3. **Recommend**: Provide personalized, actionable advice to students/faculty.

## Slide 4: System Architecture
- **Input**: Academic (Marks, Attendance) + Behavioral (Stress, Sleep) data.
- **Process**:
    - Data Cleaning & Scaling.
    - ML Model (XGBoost/Random Forest).
    - XAI Engine (SHAP Values).
- **Output**: Prediction + Visual Explanations + Recommendations.
- *(Include Level 1 DFD or ML Pipeline Diagram here)*

## Slide 5: Methodology - Data & Features
- **Dataset**: Synthetic dataset (N=1000).
- **Key Features**:
    - **Academic**: Attendance, Internal Marks, CGPA, Lab Performance.
    - **Behavioral**: Study Hours, Sleep Hours, Stress Level, Exam Anxiety.
- **Target**: Pass / Fail.

## Slide 6: Methodology - ML Models
- **Models Tested**:
    - Logistic Regression (Baseline)
    - Random Forest
    - Support Vector Machine (SVM)
    - XGBoost (Selected Best Model)
- **Evaluation Metrics**: Accuracy, F1-Score, ROC-AUC.

## Slide 7: Explainable AI (XAI)
- **Why XAI?**: Trust and Actionability.
- **Technique Used**: SHAP (SHapley Additive exPlanations).
- **Global Explanation**: "Attendance and Internal Marks are the top predictors overall."
- **Local Explanation**: "Student X is at risk specifically due to High Stress and Low Study Hours."

## Slide 8: Results & Demo
- **Model Accuracy**: ~90-95% (on synthetic data).
- **Web App**: Streamlit interface demonstrating real-time prediction.
- *(Show screenshots of the App: Input form, Prediction, Force Plot)*

## Slide 9: Conclusion & Future Scope
- **Conclusion**: The system successfully bridges the gap between prediction and intervention.
- **Future Work**:
    - Real-time integration with University LMS.
    - Mobile App for students.
    - Adding socio-economic factors.

## Slide 10: Q&A
- Thank You!
- Questions?
