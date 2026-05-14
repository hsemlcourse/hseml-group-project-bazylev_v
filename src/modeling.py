import pandas as pd
import joblib
import optuna
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import os

def train_and_optimize():
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')

    X_train, y_train = train.drop('class', axis=1), train['class']
    X_val, y_val = val.drop('class', axis=1), val['class']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr = LogisticRegression(random_state=42)
    lr.fit(X_train_scaled, y_train)

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'random_state': 42
        }
        model = XGBClassifier(**param)
        model.fit(X_train_scaled, y_train)
        return f1_score(y_val, model.predict(X_val_scaled), average='macro')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    best_xgb = XGBClassifier(**study.best_params)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    stacking_model = StackingClassifier(
        estimators=[('xgb', best_xgb), ('rf', rf)],
        final_estimator=LogisticRegression()
    )
    
    stacking_model.fit(X_train_scaled, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(stacking_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print(f"Оптимизация завершена. Лучший F1: {study.best_value}")

if __name__ == "__main__":
    train_and_optimize()
