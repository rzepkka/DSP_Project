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

# DICOM and NIfTI processing
import pydicom 
from skimage import io
import SimpleITK as sitk
import dicom2nifti
import nibabel as nib
from celluloid import Camera
from IPython.display import HTML

# Models
from classifier import classifier

timestr = time.strftime("%Y%m%d-%H%M%S")

@st.cache
def load_image(image_file):
	img = Image.open(image_file)
	return img

def text_downloader(raw_text):
	b64 = base64.b64encode(raw_text.encode()).decode()
	new_filename = "notes_file_{}_.txt".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/txt;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)


def run_models_app():

	# submenu_model = st.sidebar.selectbox("Choose Model:",["<select>","VGG 16"])

	methods = ["Saliency", "Integrated Gradients", "Deep Lift", "Grad-Cam","Occlusion"]

	preds = {"NonDemented": "Non Demented", "VeryMildDemented": "Very Mild Demented", "MildDemented": "Mild Demented", "ModerateDemented": "Moderate Demented"}
	
	explanations = {"Saliency":"Saliency is a metric for determining how significant a change in image intensity is. It is a real-valued quantity that expresses the 'strength' of an intensity variation.", 
					"Integrated Gradients":"Integrated Gradient (IG) is a deep neural network interpretability or explainability technique that visualizes the importance of the model's input features that contribute to its prediction. This method computes the gradient of the model's prediction output to its input features and does not require any changes to the deep neural network's original architecture.",
					"Deep Lift":"DeepLIFT (Deep Learning Important FeaTures) is a method for decomposing the output prediction of a neural network on a specific input by backpropagating the contributions of all neurons in the network to every feature of the input. DeepLIFT compares the activation of each neuron to its ‘reference activation’ and assigns contribution scores according to the difference.", 
					"Grad-Cam":"Gradient-weighted Class Activation Mapping (Grad-CAM) uses the gradients of any target concept flowing into the final convolutional layer to produce a coarse localization map highlighting the important regions in the image for predicting the concept.",
					"Occlusion":"Occlusion techniques in computer vision block a portion of an image during training time, challenging the network to learn not to rely canonical features. Looking at the Class Activation Map (CAM) it might seen that the network relies heavily on some particular parts of the image to make predictions."}

	st.sidebar.subheader("Machine Learning model used: VGG 16")

	chosen_method = st.sidebar.selectbox("Choose Explainable Method:",("<select>","Saliency", "Integrated Gradients", "Deep Lift", "Grad-Cam","Occlusion"))
	button_run_model = st.sidebar.button('Run Model',key='button-run-model')

	st.subheader("Explainability for Machine Learning Predictions")
	image_file = st.file_uploader("Upload an Image",type=["png","jpg","jpeg"])

	col1, col2 = st.columns(2)

	submenu_model = "VGG 16"

	# with col1:
	# 	st.subheader("Uploaded File")

	# if submenu_model in ["VGG 16","Something else"] and chosen_method in methods:
	if chosen_method in methods:
		if image_file is not None:
			im_pil = load_image(image_file)
			im_pil_rgb = im_pil.convert('RGB')

			height, width = im_pil.size
			im_pil = im_pil.resize((2*height,2*width))

			st.sidebar.info(explanations[chosen_method])


			if submenu_model == "VGG 16":
				model = classifier.load_checkpoint()
				model_prediction = classifier.make_prediction(model, im_pil_rgb)

			# LINK ANOTHER MODEL HERE
			else: # submenu_model == "VGG 16":
				model = classifier.load_checkpoint()
				model_prediction = classifier.make_prediction(model, im_pil_rgb)

			
			with col1:
				
					# st.image(load_image(im_pil)) #width=250,height=250)
					st.subheader("Uploaded File")
					st.image(im_pil)
					notes = st.text_area('Notes', height=200)
					button_save_notes = st.button('Save', key='button-save-notes')

			if button_run_model:
				st.sidebar.success("Model is running")

				with col2:
					st.subheader(f"Prediction: {preds[model_prediction]}")	
					model = classifier.load_checkpoint()			
					method = classifier.explain_image(im_pil_rgb, model_prediction, model,method=chosen_method)
					# fig.imshow(method)
					st.pyplot(method, height=100, width=100)
					st.success("The more intense the color, the more important was the feature for the prediciton.")

			else:
				st.sidebar.warning("Run the model to see the prediciton.")

			if button_save_notes:
				with col1:
					# st.write(notes)
					text_downloader(notes)
				with col2:
					st.subheader(f"Prediction: {model_prediction}")	
					method = classifier.explain_image(im_pil_rgb, model_prediction, model,method=chosen_method)
					st.pyplot(method, height=height+200, width=width+200)
					st.success("The more intense the color, the more important was the feature for the prediciton.")
		
		else:
			st.sidebar.info(explanations[chosen_method])

			st.sidebar.warning('Please upload a file')

	else:
		st.sidebar.warning('Please select explainable method to use')