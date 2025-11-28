# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# 1. Initialize the App
app = FastAPI(title="Churn Prediction API")

# 2. Define the Input Data Schema
# We list the most important raw features here.
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    # We can add defaults for categorical data to avoid errors
    Contract: str = "Month-to-month"
    PaymentMethod: str = "Electronic check"

# 3. Load the Model & Column Names on Startup
model = joblib.load("models/model.pkl")

# We need the exact column names the model was trained on.
# In a real system, we'd save this list during training. 
# For now, we load a sample of the training data to get the columns.
model_columns = pd.read_csv("data/processed/train.csv").drop("Churn", axis=1).columns

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    try:
        # Convert JSON input to a DataFrame
        input_df = pd.DataFrame([data.dict()])

        # Process: One-Hot Encoding (converting text to numbers)
        input_df = pd.get_dummies(input_df)

        # CRITICAL STEP: Align columns with the model
        # If 'PaymentMethod_Mailed check' is missing in input, add it as 0
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Make Prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1] # Probability of Churn (1)

        result = "Churn" if prediction[0] == 1 else "No Churn"

        return {
            "prediction": result,
            "churn_probability": float(probability)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)