from fastapi import FastAPI
from churn_guard.infer import Predictor
from churn_guard.schema import PredictRequest, PredictResponse


app = FastAPI(title="ChurnGuard API")
predictor = Predictor()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return predictor.predict_one(req.model_dump())
