# System Architecture & Diagrams

## 1. Data Flow Diagram (DFD)

### Level 0 DFD (Context Diagram)
```mermaid
graph LR
    User[User (Faculty/Student)] -- Input Student Data --> System[XAI Failure Prediction System]
    System -- Prediction & Explanation --> User
    System -- Recommendations --> User
```

### Level 1 DFD
```mermaid
graph TD
    User[User] -->|Input Data| Input[Data Input Module]
    Input -->|Raw Data| Preproc[Preprocessing & Cleaning]
    Preproc -->|Cleaned Features| Model[ML Prediction Model]
    Model -->|Probability| Decision{Pass/Fail?}
    Decision -->|Result| Output[Output Interface]
    Model -->|Feature Importance| XAI[XAI Engine (SHAP/LIME)]
    XAI -->|Explanations| Output
    XAI -->|Key Factors| RecSys[Recommendation Engine]
    RecSys -->|Tips| Output
```

## 2. UML Diagrams

### Use Case Diagram
- **Actors**: Faculty, Student, Administrator.
- **Use Cases**:
    - Login/Authenticate.
    - Input Student Details.
    - View Prediction (Pass/Fail).
    - View Explanation (Why?).
    - View Recommendations.
    - Train/Retrain Model (Admin only).

### Class Diagram
- **Student**: `id`, `name`, `attributes[]`, `getDetails()`
- **Predictor**: `model`, `loadModel()`, `predict(student)`
- **Explainer**: `shap_explainer`, `explain(model, student)`
- **Recommender**: `rules`, `suggest(reasons)`

## 3. Entity Relationship (ER) Diagram
Since this is primarily an ML application, the database schema is simple if we were to store data.
- **Student_Table**: `StudentID (PK)`, `Name`, `Attendance`, `InternalMarks`, `StudyHours`, `...`, `Target (Pass/Fail)`
- **Prediction_Log**: `LogID (PK)`, `StudentID (FK)`, `Prediction`, `Timestamp`

## 4. ML Pipeline Diagram
```mermaid
flowchart LR
    A[Raw Data] --> B[Data Cleaning]
    B --> C[EDA & Visualization]
    C --> D[Feature Engineering]
    D --> E[Train-Test Split]
    E --> F[Model Selection (LR, RF, XGB, SVM)]
    F --> G[Model Training]
    G --> H[Evaluation (Acc, F1, AUC)]
    H --> I[Best Model Selection]
    I --> J[XAI Analysis (SHAP)]
    J --> K[Deployment (Streamlit)]
```
