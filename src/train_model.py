import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# Ensure directories exist
os.makedirs('models', exist_ok=True)

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Separate features and target
    X = df.drop('Target_Fail', axis=1)
    y = df['Target_Fail']
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_scaled_df, y, scaler

def train_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42)
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    
    print("\nModel Evaluation Metrics:")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
    print("-" * 70)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {'Accuracy': acc, 'F1': f1, 'AUC': auc}
        
        print(f"{name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {auc:<10.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    return best_model, best_model_name

def save_artifacts(model, model_name):
    joblib.dump(model, 'models/best_model.pkl')
    print(f"Saved {model_name} as models/best_model.pkl")


if __name__ == "__main__":
    # 1. Load Data
    df = load_data('data/student_performance.csv')
    
    # 2. Preprocess
    X, y, scaler = preprocess_data(df)
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Train & Evaluate
    best_model, best_name = train_evaluate_models(X_train, X_test, y_train, y_test)
    
    # 5. Save Model
    save_artifacts(best_model, best_name)
    
    # 6. Explain (Optional run here to verify)
    # explain_model(best_model, X_train, X_test)
