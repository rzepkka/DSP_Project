import streamlit as st
import streamlit.components as stc
import streamlit.components.v1 as components

# File Processing Pkgs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import plotly.express as px
from PIL import Image

from pathlib import Path
import base64
import time

# App files
from home_app import run_home_app
from patient_app import run_patient_app
from medical_app import run_medical_app
from models_app import run_models_app


def main():
	st.set_page_config(layout="wide")
	st.title("DSP - Alzheimer's Detection Group")

	menu = ["Home", "Upload Image for Analysis","Patients Data","Medical Image Viewer","About"]
	st.sidebar.header('App Navigator')
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Home":
		run_home_app()

	elif choice == "Upload Image for Analysis":

		run_models_app()

	elif choice == "Patients Data":
		run_patient_app()

	elif choice == "Medical Image Viewer":
		run_medical_app()

	else:
		st.subheader("")
		st.subheader('?? Suggestions ??')
		st.subheader('?? Something more ??')
	

if __name__ == '__main__':
	main()