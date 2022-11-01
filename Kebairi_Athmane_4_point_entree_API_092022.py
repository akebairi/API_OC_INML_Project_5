"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests


def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {"data": data}
    response = requests.request(
        method="POST", headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

MLFLOW_URI = "http://127.0.0.1:5000/invocations"
data = np.random.rand(1,512).tolist()
pred = request_prediction(MLFLOW_URI, data)[0]
print(pred)
"""
MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

st.title('Please submit your question')

title = st.text_input('Title of the question', '')

body = st.text_area('Body of the question', '''''')


if st.button('Keywords'):
    
    data = ['android is best']
    pred = request_prediction(MLFLOW_URI, data)[0]
    
    st.write('X '*np.random.randint(1, 5))
    
else:
    st.write('')
"""