# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 21:20:24 2026

@author: Krish
"""

import json
import requests

# paste your deployed URL here
url = "http://127.0.0.1:8000/diabetes-prediction"
url = "https://barbara-unconspired-thusly.ngrok-free.dev/diabetes-prediction"
url = "https://disease-prediction-api-2cb3.onrender.com/diabetes-prediction"

input_data = {
    'Pregnancies'             :6,
    'Glucose'                 :148,
    'BloodPressure'           :72,
    'SkinThickness'           :35,
    'Insulin'                 :0,
    'BMI'                     :33.6,
    'DiabetesPedigreeFunction':0.627,
    'Age'                     :50
}

input_json = json.dumps(input_data)

response = requests.post(url, data=input_json)
print(response.text)
