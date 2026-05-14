import pandas as pd
import os

def test_data_existence():
    """тест на файлыы"""
    assert os.path.exists('data/processed/train.csv')
    assert os.path.exists('data/processed/test.csv')

def test_feature_count():
    """тест на новые фичи"""
    df = pd.read_csv('data/processed/train.csv')
    assert 'u-g' in df.columns