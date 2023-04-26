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
image = Image.open('sales.png')

# Show the image
st.image(image, caption='Sales Forecasting', use_column_width=True)



# Set up sidebar
st.sidebar.header('Navigation')
menu = ['Home', 'About']
choice = st.sidebar.selectbox("Select an option", menu)

# Home section
if choice == 'Home':
    st.markdown("<h1 style='text-align: center;'>Welcome</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>This is a simple sales prediction app.</p>", unsafe_allow_html=True)


# About section
elif choice == 'About':
    st.title('About')
    st.write('This is the about page.')
    st.write('This WebApp was designed using Streamlit.')


# Set Page Title
st.title('Sales Prediction App')
st.markdown('Select your features and click on Submit')


# Loading Machine Learning Objects
@st.cache_data()
def load_ml_objects(file_path = 'ML_toolkit'):
    # Function to load ml objects
    with open('ML_toolkit', 'rb') as file:
        loaded_object = pickle.load(file)
        
    return loaded_object


# Instantiating ML_items
loaded_object = load_ml_objects(file_path = 'ML_toolkit')

scaler = loaded_object["scaler"]
model = loaded_object["model"]
encode  = loaded_object["encoder"]
data = loaded_object["data"]


data_ = data.drop('sales', axis=1)

# Set Min and Max date interval
min_date = datetime.date(2000, 1, 1)
max_date = datetime.date(2099, 12, 31)


# Create a form for collecting input data
st.header("Input Data")

# Define the column layout
col1, col2, col3 = st.columns(3)

# Define the input fields for each column
with col1:
    date = st.date_input("Select a date", min_value=min_date, max_value=max_date, key="my_date_picker")
    family = st.selectbox("Family", options=(list( data_['family'].unique())))
    transactions = st.slider("Transactions", min_value=1, max_value=10000, step=1)

with col2:
    city = st.selectbox("City", options =(data_['city'].unique()))
    cluster = st.selectbox("Cluster", options=(list( data_['cluster'].unique())))
    store_nbr = st.slider("Store Number", min_value=1, max_value=100, step=1)
    
with col3:
    holiday_type = st.selectbox("Day Type", options =(data_['holiday_type'].unique()))
    onpromotion = st.selectbox("On Promotion", options=(list( data_['onpromotion'].unique())))
    oil_price = st.number_input("Oil Price", min_value=1, max_value=110, step=1, label_visibility='visible')

# Print the input data to the console
st.header("This Is Your Data")
st.write("Date:", date)
st.write("Family:", family)
st.write("Transactions:", transactions)
st.write("City:", city)
st.write("Cluster:", cluster)
st.write("Store Number:", store_nbr)
st.write("Day Type:", holiday_type)
st.write("On Promotion:", onpromotion)
st.write("Oil Price:", oil_price)

df_from_input = pd.DataFrame([{
   'date' : date,
   'family': family,
   'transactions': transactions,
   'city': city,
   'cluster': cluster,
   'store_nbr': store_nbr,
   'holiday_type': holiday_type,
   'onpromotion': onpromotion,
   'oil_price': oil_price
  }])


df_from_input['date'] = pd.to_datetime(df_from_input['date'])
df_from_input["year"] = df_from_input['date'].dt.year
df_from_input["month"] = df_from_input['date'].dt.month


def predict_sales(df_from_input):
    # features to encode
    categoricals = data_[["family", "city", "holiday_type"]]

    columns = list(data_.columns) 
    
    # features to scale
    numericals = data_.select_dtypes(include='number')
    
    # Scaling the columns
    scale_numericals = scaler.transform(numericals)

    # Encoding the categoricals
    encoded_categoricals = encode.transform(categoricals)
	

	# concatenate the two DataFrames
    final_data = np.concatenate([scale_numericals, encoded_categoricals], axis=1)

	

    prediction = model.predict(final_data)
    # final_data["sales"] = prediction
    # data["sales"] = prediction

    return prediction

    print(final_data)
    print(scale_numericals.shape)
    print(encoded_categoricals.shape)

    print(type(scale_numericals))
    print(type(encoded_categoricals))
 

# Prediction
if st.button('Submit'):
    # Convert the date to a Unix timestamp
    date = time.mktime(date.timetuple())

    prediction = predict_sales(df_from_input)
    # prediction(data_, df_from_input)
    st.success(' Predicted Sales is : ' + str(round(prediction[0],2)))
    # st.success('Sales is : ', round(prediction[0],2))
    # st.success(f'Sales is : {prediction}')


