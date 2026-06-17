from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import joblib
import json
from monitoring import log_prediction

# App banao
app = FastAPI(title="Customer Churn Predictor API")

# Pipeline aur features load karo
pipeline = joblib.load('pipeline.pkl')
with open('feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Input validation model
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PaperlessBilling: int
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    InternetService_Fiber_optic: int = 0
    InternetService_No: int = 0

    # Validation
    @validator('tenure')
    def tenure_valid(cls, v):
        if v < 0 or v > 72:
            raise ValueError('Tenure 0-72 ke beech hona chahiye')
        return v

    @validator('MonthlyCharges')
    def charges_valid(cls, v):
        if v < 0 or v > 200:
            raise ValueError('Monthly Charges 0-200 ke beech hona chahiye')
        return v

# Root endpoint
@app.get("/")
def home():
    return {"message": "Customer Churn Predictor API chal raha hai! ✅"}

# Predict endpoint
@app.post("/predict")
def predict(data: CustomerData):
    try:
        # Input dict banao
        input_dict = {feature: 0 for feature in feature_names}

        input_dict['tenure'] = data.tenure
        input_dict['MonthlyCharges'] = data.MonthlyCharges
        input_dict['TotalCharges'] = data.TotalCharges
        input_dict['SeniorCitizen'] = data.SeniorCitizen
        input_dict['Partner'] = data.Partner
        input_dict['Dependents'] = data.Dependents
        input_dict['PaperlessBilling'] = data.PaperlessBilling
        input_dict['Contract_One year'] = data.Contract_One_year
        input_dict['Contract_Two year'] = data.Contract_Two_year
        input_dict['InternetService_Fiber optic'] = data.InternetService_Fiber_optic
        input_dict['InternetService_No'] = data.InternetService_No

        # DataFrame banao
        input_df = pd.DataFrame([input_dict])

        # Predict karo
        prediction = pipeline.predict(input_df)[0]
        probability = pipeline.predict_proba(input_df)[0][1]

        # Log karo
        log_prediction(
            input_data=data.dict(),
            prediction=int(prediction),
            probability=round(float(probability) * 100, 2)
        )

        return {
            "prediction": int(prediction),
            "result": "CHURN" if prediction == 1 else "STAY",
            "churn_probability": round(float(probability) * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Logs dekho endpoint
@app.get("/logs")
def get_logs():
    try:
        with open('predictions_log.json', 'r') as f:
            logs = json.load(f)
        return {"total_predictions": len(logs), "logs": logs}
    except:
        return {"total_predictions": 0, "logs": []}