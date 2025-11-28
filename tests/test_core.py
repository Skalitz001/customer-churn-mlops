# tests/test_core.py
import os
import pandas as pd
import pytest

# 1. Test if the processed data exists
def test_data_files_exist():
    assert os.path.exists("data/processed/train.csv"), "Training data not found"
    assert os.path.exists("data/processed/test.csv"), "Test data not found"

# 2. Test if the data has the right shape
def test_data_columns():
    df = pd.read_csv("data/processed/train.csv")
    # Check if 'Churn' column exists (it's our target)
    assert "Churn" in df.columns, "Target column 'Churn' missing from data"

# 3. Test if the model was saved
def test_model_exists():
    assert os.path.exists("models/model.pkl"), "Model file not found"