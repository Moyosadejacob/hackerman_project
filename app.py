## This is going to be a Streamlit App

import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor


st.title('sentiment Prediction')
friends = st.number_input('author.properties.friends', 0,2000)
status_count = st.number_input('author.properties.status_count', 0,50000)
verified = st.selectbox('author.properties.verified', options=['True', 'False'])
body = st.selectbox('content.body', options=["Can't believe I'm missing Love Island 😩", 'How many times does he wonna say the phrase "i deal with shit" #LoveIsland', "@Amyyy14 thank u so much Amy you really get me ❤️ I come home tmrw let's get drinks", "@sickkening Yep you're also that xx"])
country = st.selectbox('location.country', options=['GB', 'GG', 'IM', 'JE'])
platform = st.selectbox('properties.platform', options=['twitter'])
latitude = st.number_input('location.latitude', 0,100)
longitude = st.number_input('location.longitude', -5,5)
sentiments=st.form('Sentiments prediction', clear_on_submit=True)


input_data_dict = {
    
    "author.properties.friends": friends,
    "author.properties.status_count": status_count,
    "author.properties.verified": verified,
    "content.body": body,
    "location.country": country,
    "properties.platform": platform,
    "location.latitude": latitude,
    "location.longitude": longitude
    
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