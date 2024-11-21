import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# loading the trained model
model = tf.keras.models.load_model('model.h5')

# loading scaler and encoder
with open('geo_ohencoder.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# streamlit app
st.title('Customer churn prediction')

# creating user input for streamlit app
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 62)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit score')
estimated_sal = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of products', 1, 4)
has_crd_card = st.selectbox('Has credit card', [0,1])
is_active_member = st.selectbox('Is active member', [0,1])

# putting input data into the dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_crd_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_sal],
})

# adding geo data
geo_encoded = label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# concatinating with input data
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# scaling the data
input_data_scaled = scaler.transform(input_data)

# predicting the churn 
prediction = model.predict(input_data_scaled)
probability = prediction[0][0]

st.write(f'Probability that the customer will churn is {probability:.2f}')

if probability > 0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer probably wont churn')