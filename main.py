import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#load imdb dataset word index
word_index=imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model("simple_rnn_imdb.h5")
#function to decode reviews
def decode_review(encoded_review):
  return " ".join([reverse_word_index.get(i-3,'?') for i in encoded_review])

  #function to preprocess user input
max_features = 1000
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = []

    for word in words:
        index = word_index.get(word, 2)
        if index < max_features:     # ✅ restrict vocabulary
            encoded_review.append(index + 3)
        else:
            encoded_review.append(2)  # OOV token

    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

#prediction function
def predict_sentiment(review):
  preprocessed_input=preprocess_text(review)

  prediction=model.predict(preprocessed_input)

  sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

  return sentiment,prediction

import streamlit as st
st.title("IMDB movie title review sentiment analysis")
st.write("Enter a movie review to classify it as positive or negative")

user_input =  st.text_area("Movie review")

if st.button("clasify"):
   perprocessed_input = preprocess_text(user_input)

   prediction = model.predict(perprocessed_input)

   sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"

   st.write(f"Sentiment: {sentiment}")
   st.write(f"Prediction score : {prediction[0][0]}")

else:
   st.write("Please enter a movie review")