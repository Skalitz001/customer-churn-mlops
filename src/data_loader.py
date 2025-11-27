# src/data_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

def load_and_clean_data(file_path):
    """Loads raw data, fixes types, and encodes categorical variables."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # 1. 'TotalCharges' is read as an object because of blank spaces. Fix it:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)

    # 2. 'Churn' is Yes/No. We need 1/0 for the model.
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # 3. Drop CustomerID (it's unique and useless for prediction)
    df.drop('customerID', axis=1, inplace=True)

    # 4. Simple One-Hot Encoding for other categorical columns
    # (In a real massive project, we'd use a saved encoder, but this works for now)
    df = pd.get_dummies(df)
    
    return df

def split_and_save(df, output_dir):
    """Splits data into train/test and saves them."""
    print("Splitting data...")
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"Success! Data saved to:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")

if __name__ == "__main__":
    # This block allows us to run the script from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/churn.csv")
    parser.add_argument("--output", default="data/processed")
    args = parser.parse_args()

    clean_df = load_and_clean_data(args.input)
    split_and_save(clean_df, args.output)