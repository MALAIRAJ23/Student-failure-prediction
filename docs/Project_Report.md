# Project Report: Explainable ML System to Predict Why a Student Fails

## 1. Problem Statement
In educational institutions, student failure is a critical issue that affects not only the students' career prospects but also the institution's reputation. Traditional methods of identifying at-risk students often rely on mid-term marks or attendance alone, which may be too late for effective intervention. Moreover, even when a prediction is made, the *reasons* behind the potential failure are often opaque (black-box models), making it difficult for counselors and faculty to provide targeted remedial actions. There is a need for a system that not only predicts failure with high accuracy but also explains *why* a specific student is at risk, enabling personalized and timely interventions.

## 2. Objectives
- **Primary Objective**: To develop a Machine Learning model that accurately predicts whether a student will pass or fail based on academic and behavioral attributes.
- **Secondary Objective**: To integrate Explainable AI (XAI) techniques (SHAP/LIME) to interpret model predictions and identify key contributing factors for each student.
- **Tertiary Objective**: To build a recommendation engine that suggests personalized improvements for at-risk students and a user-friendly web interface for stakeholders.

## 3. Scope
- **Target Audience**: Faculty, Academic Counselors, and Students.
- **Data**: Synthetic dataset representing student demographics, academic performance (internal marks, lab scores), and behavioral metrics (study hours, sleep, stress).
- **Techniques**: Supervised Learning (Classification), Model Interpretability (XAI).
- **Deliverables**: A functional web application, trained models, and comprehensive documentation.

## 4. Proposed System
The proposed system is an "Explainable Student Failure Prediction System". It leverages historical student data to train a predictive model. Unlike traditional systems, it incorporates an XAI layer that breaks down the prediction into understandable reasons (e.g., "Low Attendance" or "High Exam Anxiety").
- **Input**: Student details (Attendance, Marks, Stress levels, etc.).
- **Process**: Data Preprocessing -> Feature Engineering -> Model Prediction -> XAI Explanation.
- **Output**: Pass/Fail Probability, Top Failure Reasons, and Actionable Recommendations.

## 5. Literature Survey (Simulated)
| Author (Year) | Title | Methodology | Limitations |
| :--- | :--- | :--- | :--- |
| Smith et al. (2021) | "Early Warning Systems in Higher Ed" | Logistic Regression on LMS data. | Low accuracy; lacked behavioral features. |
| Johnson & Lee (2022) | "Predicting Student Dropout using RF" | Random Forest with demographic data. | Black-box model; no specific reasons provided for individual students. |
| Gupta et al. (2023) | "XAI in Education" | SVM with LIME for small datasets. | Limited scope; did not include psychological factors like stress/anxiety. |
| **Proposed Work** | **XAI-Failure Prediction Model** | **Ensemble Models (XGBoost/LGBM) + SHAP/LIME** | **Combines academic & behavioral data with global/local explanations.** |

## 6. System Requirements
### Hardware
- **Processor**: Intel Core i5 or equivalent (min. 4 cores).
- **RAM**: 8 GB or higher.
- **Storage**: 500 MB free space.

### Software
- **OS**: Windows 10/11, Linux, or macOS.
- **Language**: Python 3.8+.
- **Libraries**: Scikit-learn, Pandas, NumPy, SHAP, LIME, Streamlit, XGBoost.
- **IDE**: VS Code / Jupyter Notebook.
