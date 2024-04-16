from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from models.requests import Request
from http import HTTPStatus
import joblib
import numpy as np


detector = APIRouter()



@detector.post("/predict/forecast")
async def predict(request: Request):
    try:
        model_fit = joblib.load('./detector-model/modelo_previsao.joblib')
        forecast = model_fit.predict(request.numberofdays)
        return {"prediction": forecast}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

    
@detector.post("/predict/forecast/by-month")
async def predict(request: Request):
    try:
        if request.numberofdays != 120:
            return JSONResponse(status_code=400, content={"error": "O parâmetro 'numberofdays' deve ser igual a 120."})

        model_fit = joblib.load('./detector-model/modelo_previsao.joblib')
        forecast = model_fit.predict(request.numberofdays)
 
        monthly_forecast = []
        for i in range(0, len(forecast), 30):  
            month_forecast = forecast[i:i+30]  
            monthly_total = np.sum(month_forecast)  #
            monthly_forecast.append(monthly_total)

        return {"monthly_forecast": monthly_forecast}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



@detector.get("/accuracy")
async def accuracy():
    try:
        mape_value = 0.0216
        return {"mape_accuracy": mape_value,
                "accuracy": "98%"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
