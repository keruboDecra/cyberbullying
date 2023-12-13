import streamlit as st
import re
from nltk.corpus import stopwords
import joblib
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import nltk
from keras.preprocessing import image
import numpy as np
# Load the trained Random Forest Classifier
rf_classifier = joblib.load('random_forest_model.joblib')

# Load the TfidfVectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Load the offensive words from the en.txt file
with open('en.txt', 'r') as file:
    offensive_words = set(word.strip().lower() for word in file.readlines())

# Function to check for offensive words
def check_offensive_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    offensive_detected = [word for word in words if word in offensive_words]
    return offensive_detected

# Streamlit app
def main():
    st.title('Cyberbullying Detection App')

    # User input
    user_text = st.text_area('Enter your tweet here:', '')

    if user_text:
        # Check for offensive words
        offensive_detected = check_offensive_words(user_text)

        if offensive_detected:
            st.warning(f"Offensive word(s) detected: {', '.join(offensive_detected)}")

        # Vectorize the user input using TfidfVectorizer
        user_text_tfidf = tfidf_vectorizer.transform([user_text])

        # Predict cyberbullying
        prediction = rf_classifier.predict(user_text_tfidf)

        if prediction[0] == 0:  # Not cyberbullying
            if offensive_detected:
                st.warning("Although your text is not directly cyberbullying, it contains offensive language.")
                st.write("You can modify it before submitting.")
            else:
                st.success("Your text is not detected as cyberbullying. You can submit it.")

        else:  # Cyberbullying detected
            st.error("Cyberbullying detected in your text. Please modify it before submitting.")

if __name__ == '__main__':
    main()
