# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 02:20:31 2020

@author: shadab.akhtar
"""

# -*- coding: utf-8 -*-
"""
Created on Fri April 26 12:50:04 2020

@author: shadab.akhtar
"""


import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("xgboost.pkl","rb")
classifier=pickle.load(pickle_in)


def predict_stroke(age,avg_glucose_level,bmi,gender,hypertension,heart_disease):
    input=np.array([[age,avg_glucose_level,bmi,gender,hypertension,heart_disease]]).astype(np.float64)
    prediction=classifier.predict_proba(input)
    
    return prediction
    


def main():
    global output
    st.title("Stroke Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input("age","Type Here")
    avg_glucose_level = st.text_input("avg_glucose_level","Type Here")
    bmi = st.text_input("bmi","Type Here")
    gender = st.sidebar.selectbox("gender",("Male","Female"))
    
    hypertension = st.sidebar.selectbox("hypertension",("Yes","No"))
    heart_disease = st.sidebar.selectbox("heart_disease",("Yes","No"))
    safe_html="""
          <div style="background-color:#F4D03F;padding:10px">
    <h2 style="color:white;text-align:center;">You are safe </h2>
    </div>
    """
    
    danger_html="""
          <div style="background-color:#F08080;padding:10px">
    <h2 style="color:black;text-align:center;">You are in danger </h2>
    </div>
    """
    if st.button("Predict"):
        output=predict_stroke(age,avg_glucose_level,bmi,gender,hypertension,heart_disease)
        st.success('The probability of stroke {}'.format(output))
        
        if output > 0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)
        
        
if __name__=='__main__':
    main()
    
    
    