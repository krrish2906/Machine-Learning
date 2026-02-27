# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 20:34:54 2026

@author: Krish
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelInput(BaseModel):
    Pregnancies              : int
    Glucose                  : int
    BloodPressure            : int
    SkinThickness            : int
    Insulin                  : int
    BMI                      : float
    DiabetesPedigreeFunction : float
    Age                      : int
    

diabetes_model = pickle.load(open("trained_model.sav", "rb"))

@app.post('/diabetes-prediction')
def diabetes_pred(input_parameters: ModelInput):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    preg   = input_dictionary['Pregnancies']
    gluc   = input_dictionary['Glucose']
    bp     = input_dictionary['BloodPressure']
    skinth = input_dictionary['SkinThickness']
    ins    = input_dictionary['Insulin']
    bmi    = input_dictionary['BMI']
    dpf    = input_dictionary['DiabetesPedigreeFunction']
    age    = input_dictionary['Age']
    
    input_list = [preg, gluc, bp, skinth, ins, bmi, dpf, age]
    prediction = diabetes_model.predict([input_list])
    
    if prediction[0] == 0:
        return "The person does not have Diabetes"
    else:
        return "The person has Diabetes"
    