import joblib
import numpy as np
import os

def test_model_loading():
    model_path = 'models/best_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    assert os.path.exists(model_path), "Файл модели не найден"
    assert os.path.exists(scaler_path), "Файл скалера не найден"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    dummy_data = np.random.rand(1, 12)
    dummy_data_scaled = scaler.transform(dummy_data)
    
    prediction = model.predict(dummy_data_scaled)
    
    assert len(prediction) == 1, "Модель должна возвращать одно предсказание"
    print(f"Тест пройден, предсказанный класс: {prediction[0]}")

if __name__ == "__main__":
    test_model_loading()