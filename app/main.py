# app/main.py

import os
import joblib
from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel, Field
from app.model import (
    load_model,
    train_and_save_model,
    get_available_models,
    AVAILABLE_MODELS,
    get_model_accuracy,
    MODEL_DIR 
)
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from app.utils.logger import logger
from typing import Annotated
import time
from sklearn.metrics import accuracy_score

accuracy = None
app = FastAPI()

class CustomFitRequest(BaseModel):
    X: list[list[float]]
    y: list[int]
    name: str = "logistic"

executor = ThreadPoolExecutor(max_workers=1)

model_name = "logistic"
model = load_model(model_name)

class PredictRequest(BaseModel):
    data: Annotated[
        list[list[float]],
        Field(..., min_items=1, description="2D массив чисел")
    ] 

@app.post("/fit_custom")
def fit_custom(data: CustomFitRequest):
    logger.info(f"Custom training requested for model: {data.name}")
    try:
        model_class = AVAILABLE_MODELS[data.name].__class__
        model = model_class()
        model.fit(data.X, data.y)
        
        joblib.dump(model, os.path.join(MODEL_DIR, f"{data.name}_custom.joblib"))
        acc = model.score(data.X, data.y)
        with open(os.path.join(MODEL_DIR, f"{data.name}_custom_accuracy.txt"), "w") as f:
            f.write(str(acc))
        
        return {"status": "Custom model trained", "accuracy": acc}
    except Exception as e:
        logger.error(f"Error training custom model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    logger.info("Обращение к корню сервиса")
    return {"message": "ML-сервис запущен"}

@app.post("/predict")
def predict(request: PredictRequest):
    logger.info(f"Predict called with input: {request.data}")
    start_time = time.time()
    
    preds = model.predict(request.data)
    
    duration = time.time() - start_time
    logger.info(f"Prediction took {duration:.4f} seconds")
    
    return {"predictions": preds.tolist()}

@app.post("/fit")
def fit_model(
    name: str = Query("logistic", enum=["logistic", "random_forest"]),
    max_iter: int = Query(200, ge=50, le=1000),
    n_estimators: int = Query(100, ge=10, le=500)
):
    logger.info(f"Fit requested for model: {name}")

    params = {}
    if name == "logistic":
        params["max_iter"] = max_iter
    elif name == "random_forest":
        params["n_estimators"] = n_estimators

    future = executor.submit(train_and_save_model, name, params)

    try:
        future.result(timeout=10)
        global model, model_name
        model = load_model(name)
        model_name = name
        logger.info(f"Model {name} retrained with params {params} and set as current")
        return {"status": f"Model {model_name} retrained with {params} and set as current"}
    except TimeoutError:
        logger.warning("Model training timed out")
        return {"error": "Training timed out"}

    
@app.get("/models")
def get_models():
    logger.info("Models list requested")
    
    acc = get_model_accuracy(model_name)  

    return {
        "available_models": get_available_models(),
        "current_model": model_name,
        "accuracy": acc  
    }

@app.post("/set")
def set_model(name: str):
    global model_name, model
    logger.info(f"Set model request received: {name}")
    if name not in AVAILABLE_MODELS:
        logger.error(f"Model {name} not found in available models")
        raise HTTPException(status_code=400, detail="Model not found")
    model = load_model(name)
    model_name = name
    logger.info(f"Model switched to: {name}")
    return {"status": f"Model set to {name}"}
