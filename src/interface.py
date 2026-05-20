import streamlit as st
import requests

st.set_page_config(page_title="SDSS Space Object Classifier", page_icon="🌌")

st.title("Классификация космических объектов SDSS")
st.write(
    "Введите фотометрические параметры и redshift космического объекта, "
    "чтобы определить его тип: Звезда, Галактика или Квазар"
)

API_URL = "http://localhost:8000/predict"
st.sidebar.header("Параметры объекта")
ra = st.sidebar.number_input("Правое восхождение (ra)", value=183.870, format="%.5f")
dec = st.sidebar.number_input("Склонение (dec)", value=0.072, format="%.5f")
u = st.sidebar.number_input("Фильтр u (ультрафиолетовый)", value=19.341, format="%.5f")
g = st.sidebar.number_input("Фильтр g (зеленый)", value=18.236, format="%.5f")
r = st.sidebar.number_input("Фильтр r (красный)", value=17.832, format="%.5f")
i = st.sidebar.number_input("Фильтр i (ближний ИК)", value=17.637, format="%.5f")
z = st.sidebar.number_input("Фильтр z (инфракрасный)", value=17.521, format="%.5f")
redshift = st.sidebar.number_input("Красное смещение (redshift)", value=0.0008, format="%.6f", step=0.0001)

if st.sidebar.button("Классифицировать объект"):
    payload = {
        "ra": ra, "dec": dec,
        "u": u, "g": g, "r": r, "i": i, "z": z,
        "redshift": redshift
    }
    
    with st.spinner("Запрос обрабатывается моделью"):
        try:
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                pred_class = result["prediction_class"]
                probs = result["probabilities"]
                
                st.success(f"### Результат: {pred_class}")
                st.write("#### Распределение вероятностей:")
                for c, p in probs.items():
                    st.write(f"- **{c}**: {p*100:.2f}%")
                    st.progress(p)
            else:
                st.error(f"Ошибка API (Код {response.status_code}): {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Не удалось соединиться с FastAPI. Убедитесь, что бэкенд запущен на порту 8000.")

st.info(
    "- Маленький redshift (~0.000) обычно указывает на Звезду\n"
    "- Средний redshift (~0.05 - 0.2) чаще всего указывает на Галактику\n"
    "- Высокий redshift (> 0.5) характерен для удаленных Квазаров"
)