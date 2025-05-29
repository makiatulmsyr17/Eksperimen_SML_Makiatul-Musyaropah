import pandas as pd
import os
from sklearn.model_selection import train_test_split

def process_data(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File input tidak ditemukan: {input_path}")

    # Load dataset
    df = pd.read_csv(input_path)
    print(f"Jumlah data awal: {df.shape[0]}")

    # Hapus duplikat
    df = df.drop_duplicates()
    print(f"Jumlah data setelah menghapus duplikat: {df.shape[0]}")

    # Kolom yang akan dicek outlier-nya
    columns = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    for col in columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
        else:
            print(f"Peringatan: Kolom '{col}' tidak ditemukan dan akan dilewati.")

    print(f"Jumlah data setelah menghapus outlier: {df.shape[0]}")

    if 'target' not in df.columns:
        raise ValueError("Kolom 'target' tidak ditemukan dalam dataset.")
    
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    df_clean = df.copy()
    df_clean['split'] = None
    df_clean.loc[X_train.index, 'split'] = 'train'
    df_clean.loc[X_test.index, 'split'] = 'test'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Data hasil preprocessing disimpan di: {output_path}")

    return df_clean


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    input_file = os.path.join(base_dir, '..', 'heart_raw', 'heart_raw.csv')
    output_file = os.path.join(base_dir, 'heart_preprocessing', 'heart_cleaned_split.csv')
    process_data(input_file, output_file)
