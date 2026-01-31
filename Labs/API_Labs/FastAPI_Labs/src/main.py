from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import Dict
from predict import predict_data, predict_with_probability

app = FastAPI()

class IrisData(BaseModel):
    petal_length: float
    sepal_length: float
    petal_width: float
    sepal_width: float

class IrisResponse(BaseModel):
    response: int

class IrisDetailedResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: Dict[str, float]

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=IrisResponse)
async def predict_iris(iris_features: IrisData):
    """Basic prediction endpoint - returns only the class index."""
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]
        prediction = predict_data(features)
        return IrisResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/detailed", response_model=IrisDetailedResponse)
async def predict_iris_detailed(iris_features: IrisData):
    """Detailed prediction endpoint - returns class, name, and probabilities."""
    try:
        features = [[iris_features.sepal_length, iris_features.sepal_width,
                    iris_features.petal_length, iris_features.petal_width]]
        result = predict_with_probability(features)
        return IrisDetailedResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))