# src/train.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os
import joblib

# 1. Load the processed data
print("Loading data...")
train_data = pd.read_csv("data/processed/train.csv")
X = train_data.drop("Churn", axis=1)
y = train_data["Churn"]

# 2. Define Hyperparameters
# (In a real project, these would come from params.yaml)
params = {
    "n_estimators": 100,
    "max_depth": 20,
    "random_state": 42
}

# 3. Start MLflow Run
# This block tells MLflow to listen to everything happening inside
mlflow.set_experiment("Churn_Prediction")

with mlflow.start_run():
    print("Training model...")
    # Initialize and Train
    model = RandomForestClassifier(**params)
    model.fit(X, y)

    # Predict
    y_pred = model.predict(X)

    # Calculate Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")

    # 4. Log everything to MLflow
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

    # 5. Save the model locally and to MLflow
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    
    # This saves the model file inside the MLflow UI directly
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print("Run Complete! Check MLflow UI.")