"""
# My first app
Here's our first attempt at using data to create a table:
"""
import streamlit as st
import pandas as pd
import numpy as np

st.title('Please submit your question')

title = st.text_input('Title of the question', '')

body = st.text_area('Body of the question', '''''')


if st.button('Keywords'):
    st.write('X '*np.random.randint(1, 5))
else:
    st.write('')