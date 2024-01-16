import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from src.models.predict_model import trained_model_read

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
#file paths
trained_model = r"D:\my projects\customer-satisfaction-prediction\models\model.pkl"

# Create a function to get user input
def get_user_input():
    st.sidebar.header('User Input Parameters')
    payment_sequential = st.sidebar.slider('payment_sequential')
    payment_installments = st.sidebar.slider("Payment Installments")
    payment_value = st.sidebar.slider("Payment Value" )
    price = st.sidebar.slider("Price" )
    freight_value = st.sidebar.slider("freight_value")
    product_name_length = st.sidebar.slider("Product name length")
    product_description_length = st.sidebar.slider("Product Description length")
    product_photos_qty = st.sidebar.slider("Product photos Quantity ", 0, 10, 0 )
    product_weight_g = st.sidebar.slider("Product weight measured in grams")
    product_length_cm = st.sidebar.slider("Product length (CMs)")
    product_height_cm = st.sidebar.slider("Product height (CMs)")
    product_width_cm = st.sidebar.slider("Product width (CMs)" )

    # Store a dictionary into a data frame
    user_data = {'payment_sequential': payment_sequential,
                 'payment_installments': payment_installments,
                 'payment_value': payment_value,
                 'price': price,
                 'freight_value': freight_value,
                 'product_name_lenght': product_name_length,
                 'product_description_lenght': product_description_length,
                 'product_photos_qty': product_photos_qty,
                 'product_weight_g': product_weight_g,
                 'product_length_cm': product_length_cm,
                 'product_height_cm': product_height_cm,
                 'product_width_cm': product_width_cm,
                 }
    features = pd.DataFrame(user_data, index=[0])
    return features

# Store the user input into a variable
user_input = get_user_input()

# Set a title and display the user's input
st.title('Customer Satisfaction Prediction')

# Predict the user's input and store it into a variable

if st.button("Predict", key="Customer Satisfaction", help="A stylish button"):

    predictions = trained_model_read(file = trained_model, df= user_input)
    # Display the prediction
    st.subheader('Prediction:')
    st.write("Your customer satisfaction range between(1-5): ",round(predictions[0], 2))
    st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                predictions
            ))