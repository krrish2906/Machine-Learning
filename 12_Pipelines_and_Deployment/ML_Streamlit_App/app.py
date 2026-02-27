# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu

diabetes_model = pickle.load(open("diabetes_model.sav", "rb"))
heart_model = pickle.load(open("heart_disease_model.sav", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.sav", "rb"))

with st.sidebar:
    selected = option_menu("Multiple Disease Prediction System",
                           ["Diabetes Prediction",
                            "Heart Disease Prediction",
                            "Parkinsons Disease Prediction"],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Prediction        
if selected == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies              = st.text_input("Number of Pregnancies: ")
        SkinThickness            = st.text_input("Skin Thickness value: ")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value: ")
        
    with col2:
        Glucose                  = st.text_input("Glucose level: ")
        Insulin                  = st.text_input("Insulin level: ")
        Age                      = st.text_input("Age of person: ")
        
    with col3:
        BloodPressure            = st.text_input("Blood Pressure value: ")
        BMI                      = st.text_input("BMI value: ")
        
    diabetes_diagnosis = ''
    if st.button("Diabetes Test Result"):
        prediction = diabetes_model.predict([[
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]])
        
        if prediction[0] == 1:
            diabetes_diagnosis = "The person has diabetes"
        else:
            diabetes_diagnosis = "The person does not have diabetes"
        st.success(diabetes_diagnosis)


# Heart Disease Prediction    
if selected == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age      = st.text_input("Age of person: ")
        trestbps = st.text_input("Resting Blood Pressure: ")
        restecg  = st.text_input("Resting Electrocardiographic results: ")
        oldpeak  = st.text_input("ST depression induced by exercise: ") 
        thal     = st.text_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")
        
    with col2:
        sex      = st.text_input("Sex/Gender of person: ")
        chol     = st.text_input("Serum Cholestoral in mg/dl: ")
        thalach  = st.text_input("Maximum Heart Rate achieved: ")
        slope    = st.text_input("Slope of the peak exercise ST segment: ")
        
    with col3:
        cp       = st.text_input("Chest Pain type: ")
        fbs      = st.text_input("Fasting Blood Sugar > 120 mg/dl: ")
        exang    = st.text_input("Exercise Induced Angina: ")
        ca       = st.text_input("Major vessels colored by flourosopy: ")
    
    heart_diagnosis = ''
    if st.button("Heart Disease Test Result"):
        prediction = heart_model.predict([[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]])
        
        if prediction[0] == 0:
            heart_diagnosis = "The person does not have any heart disease"
        else:
            heart_diagnosis = "The person is having heart disease"
        st.success(heart_diagnosis)

    
# Parkinson's Disease Prediction    
if selected == "Parkinsons Disease Prediction":
    st.title("Parkinsons Disease Prediction using ML")
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        RAP = st.text_input('MDVP:RAP')
        APQ3 = st.text_input('Shimmer:APQ3')
        HNR = st.text_input('HNR')
        D2 = st.text_input('D2')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        PPQ = st.text_input('MDVP:PPQ')
        APQ5 = st.text_input('Shimmer:APQ5')
        RPDE = st.text_input('RPDE')
        PPE = st.text_input('PPE')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        DDP = st.text_input('Jitter:DDP')
        APQ = st.text_input('MDVP:APQ')
        DFA = st.text_input('DFA')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        Shimmer = st.text_input('MDVP:Shimmer')
        DDA = st.text_input('Shimmer:DDA')
        spread1 = st.text_input('spread1')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        NHR = st.text_input('NHR')
        spread2 = st.text_input('spread2')
        
    parkinsons_diagnosis = ''  
    if st.button("Parkinson's Test Result"):
        prediction = parkinsons_model.predict([[
            fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
            Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE,
            DFA, spread1, spread2, D2, PPE
        ]])                          
        
        if prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"
        st.success(parkinsons_diagnosis)