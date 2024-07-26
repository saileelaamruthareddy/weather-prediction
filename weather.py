import pandas as pd
import pickle
import streamlit as st

# Load the pre-trained model (adjust the path as needed)
with open('path_to_your_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Display logo image
st.image('logo inno.jpeg')

# Set title and header
st.title('Machine Learning Models for Accurate Weather Type Prediction')
st.header('Enter below details')

# Input fields
Temperature = st.number_input("Enter the Temperature", min_value=-273.15, max_value=10000.0, step=1.0)
Humidity = st.number_input('Enter the Humidity', min_value=0, max_value=100, step=1)
Wind_Speed = st.number_input('Enter the Wind Speed', min_value=0.0, max_value=250.0, step=1.0)
Precipitation = st.number_input('Enter the Precipitation (%)', min_value=0.0, max_value=100.0, step=1.0)
Cloud_Cover = st.radio('Select the Cloud Cover', ['partly cloudy', 'clear', 'overcast'])
Atmospheric_Pressure = st.number_input('Enter the Atmospheric Pressure', min_value=0.0, max_value=1083.0, step=1.0)
UV_Index = st.number_input('Enter the UV Index', min_value=0, max_value=11, step=1)
Season = st.radio('Select the Season', ['Winter', 'Spring', 'Summer', 'Autumn'])
Visibility = st.number_input('Enter the Visibility', min_value=1.0, max_value=240.0, step=1.0)
Location = st.radio('Select the Location', ['inland', 'mountain', 'coastal'])

# Predict weather type
if st.button("Submit"):
    try:
        # Prepare data for prediction
        d = pd.DataFrame({
            'Temperature': [Temperature],
            'Humidity': [Humidity],
            'Wind Speed': [Wind_Speed],
            'Precipitation (%)': [Precipitation],
            'Cloud Cover': [Cloud_Cover],
            'Atmospheric Pressure': [Atmospheric_Pressure],
            'UV Index': [UV_Index],
            'Season': [Season],
            'Visibility': [Visibility],
            'Location': [Location]
        })

        # Make prediction
        weather = model.predict(d)[0]
        st.write(f'Predicted weather: {weather}')

        # Display corresponding image based on the prediction
        if weather == 'Rainy':
            st.image('images/rainy.jpg', width=1000)
        elif weather == 'Cloudy':
            st.image('images/cloudy.jpg', width=1000)
        elif weather == 'Sunny':
            st.image('images/sunny.jfif', width=1000)
        elif weather == 'Snowy':
            st.image('images/snowy.jpg', width=700)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
