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

# Load Images
@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
	st.set_page_config(layout="wide")
	# st.title("DSP - Alzheimer's Detection Group")

	# with st.sidebar:
	# 	st.image(load_image('images/uva_logo.png'), use_column_width  = True, )

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
		st.image(load_image('images/uva_logo.jpeg'), use_column_width  = True, )

		st.write("This application was developed as part of the MSc Information Studies course „Data System Project“. The project was offered by Amsterdam Smart Health Lab and had the objective to serve „Scalable Healthcare with Data Science“. The project’s aim is to mitigate the existing pressure from the healthcare system by focusing on ageing population.")
		st.write("Alzheimer’s disease is the leading cause of dementia and one of the greatest medical challenges of our century. Machine learning techniques, such as  Convolutional Networks, have been found to be very useful for the diagnosis of Alzheimer’s.")
		st.write("Besides all advantages, the decision performance of the model is a black box to the user. Therefore,  it is critical that professionals don’t trust those prediction blindly and have a way to examine how the decision is made. Explanations can increase users' understanding of how the system operates.")
		st.write("In this application we use 5 different explainable Methods to explain how the model suggests the specific prediction: ")
		st.write("- Saliency is a metric for determining how significant a change in image intensity is. It is a real-valued quantity that expresses the 'strength' of an intensity variation.")
		st.write("- Integrated Gradient (IG) is a deep neural network interpretability or explainability technique that visualizes the importance of the model's input features that contribute to its prediction. This method computes the gradient of the model's prediction output to its input features and does not require any changes to the deep neural network's original architecture.")
		st.write("- DeepLIFT (Deep Learning Important FeaTures) is a method for decomposing the output prediction of a neural network on a specific input by backpropagating the contributions of all neurons in the network to every feature of the input. DeepLIFT compares the activation of each neuron to its ‘reference activation’ and assigns contribution scores according to the difference.")
		st.write("- Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.")
		st.write("- Occlusion techniques in computer vision block a portion of an image during training time, challenging the network to learn not to rely canonical features. Looking at the Class Activation Map (CAM) it might seen that the network relies heavily on some particular parts of the image to make predictions.")
		st.write("")
		st.write("Team: Denice Groen, Milan de Jonge, Francisco Pereira, Julia Rzepka, Nina Spreitzer")

if __name__ == '__main__':
	main()