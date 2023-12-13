import streamlit as st
import re
import numpy as np
import nltk
import sklearn
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Load the TfidfVectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load the list of offensive words
offensive_words = set(open('en.txt').read().splitlines())

# Streamlit app
st.title("Cyberbullying Detection App")

# Input text box for user's tweet
user_input = st.text_area("Enter your tweet:")

# Check for offensive words
offensive_detected = any(word in user_input.lower() for word in offensive_words)

if offensive_detected:
    st.warning("Offensive word detected! Please modify your text.")

# Make prediction only if no offensive words are detected
elif st.button("Check for Cyberbullying"):
    # Preprocess the user input
    cleaned_input = clean_text(user_input)
    tokenized_input = cleaned_input.split()
    tokenized_input = [word.lower() for word in tokenized_input if word not in stop_words]
    tokenized_input = [lemmatizer.lemmatize(word) for word in tokenized_input]

    # Transform the input using the TfidfVectorizer
    input_tfidf = tfidf_vectorizer.transform([' '.join(tokenized_input)])

    # Make prediction
    prediction = model.predict(input_tfidf)[0]

    # Display the prediction
    if prediction == 1:
        st.error("Cyberbullying Detected! Please modify your text.")
    else:
        st.success("No Cyberbullying Detected. You can proceed.")

# Additional information for the user
st.info("If your text is classified as not cyberbullying but contains offensive language, you can modify it.")


