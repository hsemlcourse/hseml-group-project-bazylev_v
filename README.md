[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Астрономическая классификация объектов SDSS

**Студент:** Базылев Вячеслав Дмитриевич

**Группа:** БИВ236


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)

## Описание задачи

**Задача:** Мультиклассовая классификация космических объектов на три категории: звезды (STAR), галактики (GALAXY) и квазары (QSO).

**Датасет:** Sloan Digital Sky Survey (SDSS) DR14. Включает фотометрические данные (фильтры u, g, r, i, z) и параметры красного смещения (redshift).

**Целевая метрика:** **F1-macro**. Выбрана из-за возможного дисбаланса классов, чтобы одинаково качественно учитывать точность предсказания как массовых (галактики), так и редких объектов (квазары).

## Структура репозитория
.
├── README.md
├── data
│   ├── processed
│   │   ├── test.csv
│   │   ├── train.csv
│   │   └── val.csv
│   └── raw
│       ├── Skyserver_SQL2_27_2018 6_51_39 PM.csv
│       └── Skyserver_SQL2_27_2018 6_51_39 PM.csv.zip
├── models
│   ├── best_model.pkl
│   └── scaler.pkl
├── notebooks
├── presentation
│   └── README.md
├── report
│   └── report.md
├── requirements.txt
├── src
│   ├── __init__.py
│   ├── modeling.py
│   └── preprocessing.py
└── tests
    ├── __pycache__
    │   └── test.cpython-311-pytest-9.0.2.pyc
    └── test.py

## Запуск
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 src/preprocessing.py
python3 src/modeling.py
pytest tests/test.py
```
## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные

## Результаты
Модель	F1-macro	Accuracy	Примечание
Baseline (LogReg)	0.9754	0.9790	Точка отсчета
XGBoost + Optuna	0.9901	0.9920	Финальная модель (CP2)
## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)