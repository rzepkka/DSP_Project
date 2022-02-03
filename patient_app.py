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

timestr = time.strftime("%Y%m%d-%H%M%S")


def csv_downloader(data_frame,patient_id):
	csvfile = data_frame.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "patient_{}_file_{}_.csv".format(str(patient_id),timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)

	

def run_patient_app():
	submenu = st.sidebar.selectbox("Choose a method to upload data:",["Upload Patient's File","Introduce Data Manually"])

	if submenu == "Upload Patient's File":

		st.error("Ideally our application would be directly connected to a medical database to automatically link the diagnosis to an individual patient and his/her personal details. As this extension was not feasible within this course, this page was added. Here, the diagnosis can be manually linked to the patient's information.") # However, as this feature is not directly related to the main purpose of the project, viewed page will not be a part of the user testing.")

		st.subheader("Upload Patient's Data")
		data_file = st.file_uploader("Upload CSV",type=["csv"])
		if data_file is not None:
			df = pd.read_csv(data_file)
			st.dataframe(df)

	else:

		st.error("Ideally our application would be directly connected to a medical database to automatically link the diagnosis to an individual patient and his/her personal details. As this extension was not feasible within this course, this page was added. Here, the diagnosis can be manually linked to the patient's information.") # However, as this feature is not directly related to the main purpose of the project, viewed page will not be a part of the user testing.")

		st.subheader("Complete the following form with Patient's Medical data")
		st.info("As operators of this site, we take the protection of your personal information very seriously. Uploaded information will neither be stored nor disclosed to third parties.")
		
		with st.form(key='form-patient-data', clear_on_submit=True):
			firstname = st.text_input("First Name")
			surname = st.text_input("Last Name")
			_id = st.text_input("Patient's ID")
			date = st.date_input('Date of Birth')

			col1, col2 = st.columns(2)
			with col1:
				weight = st.number_input('Weight',40,150)

			with col2:
				sex = st.radio("Sex", ('F','M'))

			stage = st.selectbox("Diagnosed stage of Alzheimer's", ["Not diagnosed", "Non Demented", "Very Mild Demented", "Mild Demented", "Moderate Demented"])

			button_submit = st.form_submit_button(label='Submit Form')

			if button_submit:
				if firstname != "" and surname != "" and _id != "":
					st.success("You successfully uploaded patient's data")
					df = pd.DataFrame({'Name':str(firstname.upper()),'Surname':str(surname.upper()),"Patient's ID":str(_id),'Date of Birth':str(date), 'Sex':sex,'Weight':str(weight), 'Stage of disease':stage},index=[0])
					csv_downloader(df.T, _id)
				else:
					st.warning("Please fill in patient's data")


		st.subheader("")
