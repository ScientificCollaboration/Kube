#import tensorflow as tf

from fastapi import FastAPI
from pydantic import BaseModel
from XGBs import *
import numpy as np

##MODEL = clf_multilabel

app = FastAPI()

class UserInput(BaseModel):
    user_input: float

@app.get('/')
async def index():
    return {"Message": "This is Index"}

@app.post('/predict/')
async def predict(UserInput: UserInput):

    prediction = clf_multilabel.predict([UserInput.user_input])

    return {"prediction": float(prediction)}