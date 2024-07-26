import pandas as pd
import sweetviz
import pkg_resources
from feature_engine.outliers import Winsorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st
from sklearn.metrics import accuracy_score  # Corrected import statement


with open('weather.pkl', 'rb') as file:
    model = pickle.load(file)

st.image("logo inno.jpeg")
name=st.title('Machine Learning Models for Accurate Weather Type Prediction')
st.header('Enter below details')


Temperature =st.number_input("Enter the Temperature",min_value = -273.15, max_value=10000.0,step=1.0)
Humidity = st.number_input('Enter the Humidity',min_value=0,max_value=100,step=1)
Wind_Speed = st.number_input('Enter the Wind Speed',min_value=0.0,max_value=250.0,step=1.0)
Precipitation=st.number_input('Enter the Precipitation (%)',min_value=0.0,max_value=100.0,step=1.0)
Cloud_Cover=st.radio('Select the Cloud_Cover',['partly cloudy','clear','overcast'])
Atmospheric_Pressure = st.number_input('Enter the Atmospheric Pressure',min_value=0.0,max_value=1083.0,step=1.0)
UV_Index=st.number_input('Enter the UV Index',min_value=0,max_value=11,step=1)
Season=st.radio('Select the Season',['Winter','Spring','Summer','Autumn'])
Visibility=st.number_input('Enter the Visibility',min_value=1.0,max_value=240.0,step=1.0)
Location=st.radio('Select the Location',['inland','mountain','coastal'])
if st.button("Submit"):
    d = pd.DataFrame({
    'Temperature': [Temperature],
    'Humidity': [Humidity],
    'Wind Speed': [Wind_Speed],
    'Precipitation (%)':[Precipitation],
    'Cloud Cover': [Cloud_Cover],
    'Atmospheric Pressure': [Atmospheric_Pressure],
    'UV Index': [UV_Index],
    'Season': [Season],
    'Visibility (km)': [Visibility],
    'Location': [Location]
    })
    weather=model.predict(d)[0]
    st.write(weather)
    if weather =='Rainy':
        st.image("rainy.jpeg",width=1000)
    elif weather=='Cloudy':
        st.image("cloudy.jpeg",width=1000)
    elif weather=='Sunny':
        st.image("sunny.jpeg",width=1000)
    elif weather=='Snowy':
        st.image("snowy.jpeg",width=700)
     

     

    
