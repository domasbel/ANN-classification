import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# load the imdb reviews for word index
word_index = imdb.get_word_index()
reverse_word_index = {values: key for key, values in word_index.items()}

# loading the evaluated model
model = load_model('')

def decode_review(encoded_rev):
    return ' '.join([reverse_word_index.get(i -3, '?') for i in encoded_rev])


def preporcessed_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# prediction function 
def predict_sent(review):
    preprocessed_input = preporcessed_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

# streamlit app part

st.title('IMDB movie review classification')
st.write('This site is using simple RNN model that was trianed on IMDB movies, thus you can try it out!')


# user input
user_review = st.text_area('movie review')

if st.button('Classify'):

    preporcessed_input = preporcessed_text(user_review)

    # make prediction
    prediction = model.predict(preporcessed_input)

    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'

    # displaying the results 
    st.write(f'Sentiment that the model got is {sentiment}')
    st.write(f'This is with probability score of {prediction[0][0]}')
else:
    st.write('please enter a movie review')