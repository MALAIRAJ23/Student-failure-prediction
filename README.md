# Explainable ML System to Predict Student Failure (XAI-Failure Prediction)

## 📌 Project Overview
This project is a comprehensive Machine Learning system designed to predict student failure and, more importantly, explain *why* a student is at risk. It leverages **Explainable AI (XAI)** techniques like SHAP and LIME to provide transparent insights into the model's decision-making process, enabling educators to take targeted remedial actions.

## 🚀 Features
- **Accurate Prediction**: Uses advanced ensemble models (XGBoost, Random Forest) to predict Pass/Fail status.
- **Explainability (XAI)**:
    - **Global Explanation**: Which features matter most overall?
    - **Local Explanation**: Why is *this specific student* at risk?
- **Personalized Recommendations**: Suggests actionable improvements based on the student's specific risk factors.
- **Interactive Web App**: User-friendly Streamlit interface for real-time analysis.
- **Synthetic Data Generator**: Includes a script to generate realistic student datasets.

## 📂 Folder Structure
```
ml-project/
├── data/                   # Dataset files
│   └── student_performance.csv
├── docs/                   # Documentation & Diagrams
│   ├── Project_Report.md
│   ├── Architecture_Diagrams.md
│   ├── IEEE_Abstract.md
│   └── Presentation_Content.md
├── models/                 # Saved ML models & Scalers
│   ├── best_model.pkl
│   └── scaler.pkl
├── src/                    # Source code
│   ├── data_generator.py   # Data generation script
│   └── train_model.py      # ML training pipeline
├── app.py                  # Streamlit Web Application
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ Tech Stack
- **Language**: Python 3.8+
- **ML Libraries**: Scikit-learn, XGBoost, LightGBM, Pandas, NumPy
- **XAI Libraries**: SHAP, LIME
- **Web Framework**: Streamlit
- **Visualization**: Matplotlib, Seaborn

## ⚙️ Installation & Usage

1. **Clone the repository** (or download files):
   ```bash
   git clone <repo_url>
   cd ml-project
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate Data** (Optional, if csv not present):
   ```bash
   python src/data_generator.py
   ```

4. **Train Model**:
   ```bash
   python src/train_model.py
   ```
   *This will train multiple models, select the best one, and save it to `models/`.*

5. **Run Web App**:
   ```bash
   streamlit run app.py
   ```

## 📊 Model Performance
The system evaluates multiple models (Logistic Regression, Random Forest, SVM, XGBoost). The best model is selected based on F1-Score and AUC to ensure balanced performance between precision and recall.

## 🔮 Future Enhancements
- Integration with real-time LMS data.
- Mobile application for student self-monitoring.
- NLP analysis of student feedback.
"# Student-failure-prediction" 
