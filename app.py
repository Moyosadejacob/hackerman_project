## This is going to be a Streamlit App

import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor


st.title('sentiment Prediction')
friends = st.number_input('author.properties.friends', 0,2000)
status_count = st.number_input('author.properties.status_count', 0,2000)
verified = st.selectbox('author.properties.verified', options=['True', 'False'])
body = st.selectbox('content.body', options=["Can't believe I'm missing Love Island 😩", 'How many times does he wonna say the phrase "i deal with shit" #LoveIsland', "@Amyyy14 thank u so much Amy you really get me ❤️ I come home tmrw let's get drinks"])
country = st.selectbox('location.country', options=['GB', 'GG', 'IM', 'JE'])
platform = st.selectbox('properties.platform', options=['twitter'])
sentiment = st.selectbox("properties.sentiment", options=[ 1., -1., 0.])
latitude = st.number_input('location.latitude', 0,100)
longitude = st.number_input('location.longitude', -5,5)
sentiments=st.form('Sentiments prediction', clear_on_submit=False)


input_data_dict = {
    
    "author.properties.friends": 114,
    "author.properties.status_count": 1377,
    "author.properties.verified": "True",
    "content.body": "Can't believe I'm missing Love Island 😩",
    "location.country": 'GG',
    "properties.platform": 'twitter',
    "properties.sentiment": 1,
    "location.latitude": 52.96974444,
    "location.longitude": -1.172266
    
}
st.write(input_data_dict)

input_data=pd.DataFrame([input_data_dict])

st.write(input_data)

save_path  = "model"

save_model_predictor = TabularPredictor.load(save_path, require_version_match=False)
submit = st.button("CLICK TO PREDICT SENTIMENTS")
if submit:
    sentiments_prediction = save_model_predictor.predict(input_data)[0]
    st.write(f"Sentiments prediction value is: {sentiments_prediction}")