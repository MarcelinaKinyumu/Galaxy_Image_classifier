# -------------
# Import libraries
# -------------

import streamlit as st
import pandas as pd
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

import tensorflow as tf

## Switch off warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

## Read twitter data
path = os.path.dirname(__file__)

## Add title and subtitle to the main interface of the app
st.title("The Galaxy Classifier")

st.markdown("### Galaxy Classification")

# Hidden text that can be expanded to show explaination
with st.expander("See explanation"):
    st.markdown("Galaxies are mainly classified in two categories either as spirals or ellipticals based on their structure/morphology. The ... ")
    
image = st.container()
with image:
    #Create two columns
    col1, col2 = st.columns(2)
    
    # Display cuctomer text depending on average polarity
    with col1:
    	st.markdown("#### Spiral Galaxy")
    	st.markdown("The image shown below is a sample of a spiral galaxy")
    	image = Image.open("/home/marcelina/Desktop/DSI/project/Spiral_galaxy.jpg")
    	image = image.resize((1800,1800),Image. ANTIALIAS)
    	st.image(image, use_column_width = True)
    	#with st.expander("The image shown below is a sample of a spiral galaxy"):
    	
    
    with col2:
    	st.markdown("#### Elliptical Galaxy")
    	st.markdown("The image shown below is a sample of an elliptical galaxy")
    	image = Image.open("/home/marcelina/Desktop/DSI/project/Elliptical_galaxy.jpg")
    	image = image.resize((1800,1800),Image. ANTIALIAS)
    	st.image(image, use_column_width = True)
    	
#st.markdown("### Please upload a model below")

@st.cache(allow_output_mutation = True)
def load_model():
	model = tf.keras.models.load_model("/home/marcelina/Desktop/DSI/project/best_model2.h5")
	return model
	
with st.spinner("Loading Model into Memory..."):
	model = load_model() 
	
classes = ["Elliptical", "Spiral"] 

st.markdown(" #### Image uploader")
file = st.file_uploader("Upload a image of a galaxy", type = ["jpg", "png"])  	
    	
def import_and_predict(image_data, model):
	size = (212,212)
	#image = image_data.reshape(212,212,1)
	image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
	#image = image.convert('L') 
	img = np.asarray(image)
	img_reshape = img[np.newaxis,...]
	prediction = model.predict(img_reshape)
	
	return prediction

if file is None:
	st.markdown("Please upload an image")
	
else:
	image = Image.open(file)
	st.image(image, use_column_width = True)
	prediction = import_and_predict(image, model)
	string = classes[np.argmax(prediction)]
	st.success(string)
#Pressions, recall	
		
	
    	
    	
    	
