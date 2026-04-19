import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def prepare_data(input_path='data/raw/Skyserver_SQL2_27_2018 6_51_39 PM.csv'):
    df = pd.read_csv(input_path)
    
    df['u-g'] = df['u'] - df['g']
    df['g-r'] = df['g'] - df['r']
    df['r-i'] = df['r'] - df['i']
    df['i-z'] = df['i'] - df['z']
    
    drop_cols = ['objid', 'specobjid', 'run', 'rerun', 'camcol', 'field', 'plate', 'mjd', 'fiberid']
    df_clean = df.drop(columns=drop_cols)
    
    le = LabelEncoder()
    df_clean['class'] = le.fit_transform(df_clean['class'])
    
    train_val, test = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['class'])
    train, val = train_test_split(train_val, test_size=0.25, random_state=42, stratify=train_val['class'])
    
    os.makedirs('data/processed', exist_ok=True)
    train.to_csv('data/processed/train.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    
    print("Данные успешно подготовлены и сохранены в data/processed/")

if __name__ == "__main__":
    prepare_data()