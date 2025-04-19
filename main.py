import numpy as np
import tensorflow as tf
from keras.datasets import imdb
#from keras.preprocessing import sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i+3,'?') for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    #padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    padded_review = pad_sequences([encoded_review], maxlen=500)

    return padded_review

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]




import streamlit as st 
st.title('IMDB Movie Review Sentiment Analysis')
st.markdown('**Enter a movie review to classify it as positive or negative.**')
st.write('Note:Keep space between punctuations')
st.write('example1: good (Sentiment: Negative)')
st.write('example2:This movie was fantastic ! The acting was great and the plot was thrilling (Sentiment:Positive)')
#user input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)
    
#make prediction
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
    
else:
    st.write('Please enter a movie review. ')
    