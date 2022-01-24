import streamlit as st
import streamlit.components as stc
import streamlit.components.v1 as components

# File Processing Pkgs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import plotly.express as px
from PIL import Image
import docx2txt

from pathlib import Path
import base64
import time

# Load Images
@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

def run_home_app():
	st.image(load_image('images/home_2.png'), use_column_width  = True, )
	st.subheader("Welcome to Alzheimer's Detection Application!")
	st.write("Explainable AI provides transparency in model decisions and helps adapting ML models in contextual usage as professionals are provided with transparency by retracing the decision made. This enables to critically scrutinize the decision and builds trust.")
	st.write("The machine learning (ML) model classifies MRI images in four stages of Alzheimer. The aim of our application is to include an explainable component to show professionals on which are of the image the classifier made their decision upon. ")
	st.subheader("")
