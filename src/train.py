import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import os

def train():
    train = pd.read_csv('data/processed/train.csv')
    val = pd.read_csv('data/processed/val.csv')
    
    X_train, y_train = train.drop('class', axis=1), train['class']
    X_val, y_val = val.drop('class', axis=1), val['class']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    models = {
        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_f1 = 0
    best_model = None
    
    print("Начало экспериментов...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_val_scaled)
        score = f1_score(y_val, preds, average='macro')
        print(f"Модель {name}: F1-macro = {score:.4f}")
        
        if score > best_f1:
            best_f1 = score
            best_model = model
            
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Лучшая модель сохранена")

if __name__ == "__main__":
    train()