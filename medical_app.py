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

def run_medical_app():
	submenu = st.sidebar.selectbox("Choose file type:",["select","DICOM","NIfTI"])

	if submenu == "DICOM":

		chosen_number = st.sidebar.radio("Choose number of files to upload:",("1 file","multiple files"), key='radio-file-number')

		if chosen_number == '1 file':
			st.subheader("Upload a DICOM file for viewing and patient's data extraction")
			image_file = st.file_uploader("Upload an Image", type=["dcm"])

			if image_file is not None:

				col1, col2 = st.columns(2)

				dicom_data = {}
				dicom_file = pydicom.read_file(image_file)
				lista = [dicom_file[0x0010, 0x0020], dicom_file[0x0010, 0x0040], dicom_file[0x0010, 0x1010],dicom_file[0x0010, 0x1030],dicom_file[0x0018, 0x0020],dicom_file[0x0008, 0x103e],dicom_file[0x0018, 0x0021],dicom_file[0x0018, 0x0022],dicom_file[0x0018, 0x0023],dicom_file[0x0018, 0x0084],dicom_file[0x0018, 0x0087],dicom_file[0x0018, 0x0088]]

				for item in lista:
					d={str(item.name):str(item.value)}
					dicom_data.update(d)

				dicom_data=pd.Series(dicom_data)
				df=pd.DataFrame(dicom_data, columns=['Value'])

				ct = dicom_file.pixel_array # load the image pixel data as a numpy array
				fig = plt.figure(figsize=(4,4))
				plt.imshow(ct, cmap="gray")

				with col1:
					st.pyplot(fig, height=200, width=200)

				with col2:
					st.subheader('MRI Scanning Data')
					st.dataframe(df, width=1000,  height=1000)
					notes = st.text_area('Notes')
					if st.button('Save', key='button-save'):
						text_downloader(notes)

		elif chosen_number=="multiple files":

			chosen_output_format = st.sidebar.radio("Choose the output format:",("Grid","Slider"), key='radio-output-formats')

			st.subheader("Enter full path to the folder with all DICOM files")
			path_input = st.text_input('Input path and press Enter to apply:')


			if chosen_output_format == "Grid":

				chosen_size = st.sidebar.radio("Choose the size of the output grid:",('2x2','3x3','4x4','5x5','6x6'), key='radio-grid-size')

				size = int(chosen_size[0])				

				if path_input !='':

					try:
						path_to_head_mri = Path(str(path_input))
						series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path_to_head_mri))
						series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path_to_head_mri), series_ids[0])
						series_reader = sitk.ImageSeriesReader()
						series_reader.SetFileNames(series_file_names)
						image_data = series_reader.Execute()

						head_mri = sitk.GetArrayFromImage(image_data)

						fig, axis = plt.subplots(size, size, figsize=(10, 10))
						slice_counter = 0
						for i in range(size):
						    for j in range(size):
						        axis[i][j].imshow(head_mri[slice_counter], cmap="gray")
						        slice_counter+=1
						        # plt.axis("off")
						
						st.pyplot(fig)
					except(IndexError):
						st.error('Path not found. Cannot display images.')

			elif chosen_output_format == "Slider":

				col1, col2 = st.columns(2)

				if path_input !='':
					try:
						path_to_head_mri = Path(str(path_input))
						lenght = len(list(path_to_head_mri.glob("*")))
						st.write(f'Number of files in the path folder: {lenght}') 
						slider_file = st.slider("Choose the file number to display:",1,lenght, key='slider-output-format')


						series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path_to_head_mri))
						series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path_to_head_mri), series_ids[0])
						series_reader = sitk.ImageSeriesReader()
						series_reader.SetFileNames(series_file_names)
						image_data = series_reader.Execute()

						head_mri = sitk.GetArrayFromImage(image_data)

						fig = plt.figure(figsize=(4,4))

						plt.imshow(head_mri[slider_file], cmap="gray")
						plt.axis("off")
						plt.title(f'File number {slider_file}:')
						
						with col1:
							st.pyplot(fig,height=100, width=100)
					
						with col2:
							st.subheader('\n\n\n\n')
							st.subheader('\n\n\n\n')

							notes = st.text_area('Notes', height=180)
							if st.button('Save', key='button-save-2'):
								text_downloader(notes)	

					except(IndexError):
						st.error('Path not found. Cannot display images.')

	elif submenu == "NIfTI":

		st.subheader("Enter full path to the folder with all NIfTI files")
				
		path_input = st.text_input('Input path and press Enter to apply:')

		try:

			root = Path(str(path_input))
			lenght = len(list(root.glob("*")))

			if path_input !='':

				st.write(f'Number of files in the path folder: {lenght}') 

				subjects = [i for i in range(1,(lenght+1))]
				subjects = tuple(subjects)

				# col1, col2 = st.columns(2)

				# with col1:
				subject_chosen = st.selectbox('Choose the file number to display:', subjects, key='radio-files')

				# with col2:
				st.title(f'Video for file number {subject_chosen}\nClick "Play" to start the animation')
				sample_path = list(root.glob("*"))[int(subject_chosen)-1]
				data = nib.load(sample_path)
				ct = data.get_fdata()
				fig = plt.figure()
				camera = Camera(fig)  # Create the camera object from celluloid

				for i in range(ct.shape[2]):
					plt.imshow(ct[:,:,i], cmap='bone')
					plt.axis('off')
					camera.snap() # Store the current slice

				plt.tight_layout()
				animation = camera.animate() # Create the animation

				components.html(animation.to_jshtml(), width=1800, height=1800)

		except:
			st.error('Cannot convert to video. Please, input correct path.')

	else:
		pass

