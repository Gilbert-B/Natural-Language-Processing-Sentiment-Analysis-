# importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import os, pickle
import datetime
import time
from PIL import Image


# Set Page Configurations
st.set_page_config(page_title = "Sentiment Analysis", page_icon = ":trendline:", 
layout = "wide", initial_sidebar_state = "auto")


# Import the image
image = Image.open('sentiments.png')

# Show the image
st.image(image, caption='Sentiment Analysis', use_column_width=True)



# Set up sidebar
st.sidebar.header('Navigation')
menu = ['Home', 'About']
choice = st.sidebar.selectbox("Select an option", menu)

# Home section
if choice == 'Home':
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This App predicts the esentiments expressed in an input.</p>", unsafe_allow_html=True)


# About section
elif choice == 'About':
    st.title('About')
    st.write('This is the about page.')
    st.write('This WebApp was designed using Streamlit.')


# Set Page Title
st.title('Sentiment Prediction App')
st.markdown('Select your features and click on Submit')





