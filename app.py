import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🍷 Wine Class Prediction App")

# Input sliders
Alcohol = st.slider('Alcohol', 11.03, 14.83, 13.0)
Alcalinity_of_ash = st.slider('Alcalinity of ash', 10.0, 30.0, 19.0)
Magnesium = st.slider('Magnesium', 70, 162, 100)
Total_phenols = st.slider('Total phenols', 0.98, 3.88, 2.5)
Flavanoids = st.slider('Flavanoids', 0.38, 5.08, 2.5)
Nonflavanoid_phenols = st.slider('Nonflavanoid phenols', 0.13, 0.66, 0.3)
Color_intensity = st.slider('Color intensity', 1.28, 13.0, 5.0)
Hue = st.slider('Hue', 0.48, 1.71, 1.0)
diluted_wines = st.slider('OD280/OD315 of diluted wines', 1.27, 4.0, 2.0)

wine_classes = {
    0: "🍇 Class 1: Wines made from Cultivar 1 (Grape A)",
    1: "🍷 Class 2: Wines made from Cultivar 2 (Grape B)",
    2: "🥂 Class 3: Wines made from Cultivar 3 (Grape C)"
}

# Predict button
if st.button('Predict Wine Class'):
    input_data = np.array([[Alcohol, Alcalinity_of_ash, Magnesium, Total_phenols,
                            Flavanoids, Nonflavanoid_phenols, Color_intensity,
                            Hue, diluted_wines]])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    st.success(f"Prediction: {wine_classes[prediction]}")