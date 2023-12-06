import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from utils.feature_constructor import FeatureConstructor
from utils.loaded_model import LoadedModel
from utils.ml_api import load_model, load_feature_constructor

from utils.data_structures import PersonsList


class ModelResponse(BaseModel):
    result: PersonsList


class ConstructorResponse(BaseModel):
    result: str


model: LoadedModel
constructor: FeatureConstructor
app = FastAPI()


# Загрузка модели при старте приложение
@app.on_event("startup")
def startup_event(model_path: str = './models/RandomForest.pkl'):
    global model, constructor
    model = load_model(model_path)
    constructor = load_feature_constructor()


# create a route
@app.get("/")
def index() -> dict[str, str]:
    return {"text": "Probability predict"}


# GET - запрос для предикта. X: pd.DataFrame
@app.get("/predict")
def predict_sentiment(X):
    pred = model.predict(X)
    response = ModelResponse(result=pred.result)
    return response


# Your FastAPI route handlers go here
@app.get("/feature_construct")
def predict_sentiment(path: str = './data/test.csv'):
    constructor.feature_construct(path)
    response = 'Датасет обработан!'
    return response
