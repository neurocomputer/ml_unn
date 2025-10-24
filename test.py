import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

import os
os.environ['OPENSSL_VER'] = '1.1.1'

import streamlit as st

import numpy as np

from tensorflow.keras.models import load_model

st.header("Image Recognition App")
st.caption("Upload an image. ")
st.caption("The application will infer the one label out of 4 labels: 'Cloudy', 'Desert', 'Green_Area', 'Water'.")
st.caption("Warning: Do not click Recognize button before uploading image. It will result in error.")
