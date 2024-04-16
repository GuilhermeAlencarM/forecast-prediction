from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from models.requests import Request
from http import HTTPStatus
import joblib
import pickle

# Carregar o modelo
with open('./detector-model/model_arima.pkl', 'rb') as pkl:
    model_fit = pickle.load(pkl)


detector = APIRouter()

@detector.post("/predict/forecast")
async def predict(request: Request):
    try:
        forecast = model_fit.predict(request.numberofdays)
        return {"prediction": forecast}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@detector.get("/accuracy")
async def accuracy():
    try:
        mape_value = 0.0420
        return {"mape_accuracy": mape_value}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
