from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load your trained model
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input schema using Pydantic
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # Add all your input features here

app = FastAPI()

@app.get("/")
def home():
    return {"message": "ML Model is running!"}

@app.post("/predict")
def predict(data: InputData):
    input_data = [[
        data.feature1,
        data.feature2,
        data.feature3
        # Make sure these match the order used during training
    ]]
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}
