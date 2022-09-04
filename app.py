import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.write("""
# Rented Bike Count Prediction App
This app predicts the **Number of Bike Rented**!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():

    Hour = st.sidebar.slider('Hour',0.0,23.0,11.5)
    temperature = st.sidebar.slider('Temperature(°C)', -17.80 ,39.4, 12.88)
    humidity = st.sidebar.slider('Humidity(%)', 0.0, 98.0, 58.22)
    wind_speed = st.sidebar.slider('Wind speed (m/s)',0.0,7.4,1.72)
    visibility = st.sidebar.slider('Visibility (10m)',27.0,2000.0,1436.825)
    dew_point_temperature = st.sidebar.slider('Dew point temperature(°C)',-30.6,27.2,4.07)
    solar_radiation = st.sidebar.slider('Solar Radiation (MJ/m2)', 0.0,3.52,0.569)
    rainfall = st.sidebar.slider('Rainfall(mm)',0.0,35.0,0.1486)
    snowfall = st.sidebar.slider('Snowfall (cm)', 0.0,8.80,0.075)
    Seasons = st.sidebar.selectbox('Seasons',('Winter','Spring','Summer','Autumn'))
    Holiday = st.sidebar.selectbox('Holiday',("No Holiday","Holiday"))
    functioning_day = st.sidebar.selectbox('Functioning Day',("Yes","No"))
       
    data = {'Hour':Hour,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed ,
            'visibility': visibility ,
            'dew_point_temperature': dew_point_temperature,
            'solar_radiation': solar_radiation  ,
            'rainfall': rainfall,
            'snowfall': snowfall ,
            'Seasons': Seasons,
            'Holiday': Holiday,
            'functioning_day': functioning_day
            }
 
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Read the csv file
data_raw=pd.read_csv('SeoulBikeData.csv')
data = data_raw.drop(columns=["Rented Bike Count","Date","Unnamed: 0"],axis=1)
df= pd.concat([input_df,data],axis=0)


# Encoding
encode=["Seasons","Holiday","functioning_day"]
for col in encode: 
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df,dummy],axis = 1)
    del df[col]
df=df[:2]


# loading  Random Forest
pickle_in = open('LGBM.pkl', 'rb') 
model = pickle.load(pickle_in)


# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')


# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of Bike Count')
st.write(prediction)
st.write('---')