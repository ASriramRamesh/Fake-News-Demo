import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained model and vectorizer
@st.cache_data
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

model, vectorizer = load_model()

# Streamlit UI
st.title("Fake News Detector")

# Input fields
title = st.text_input("Enter the news title")
text = st.text_area("Enter the news text")

# Prediction function
def predict_news(title, text):
    combined_text = title + " " + text
    text_transformed = vectorizer.transform([combined_text])
    prediction = model.predict(text_transformed)
    return "Real" if prediction[0] == 1 else "Fake"

# Button to make prediction
if st.button("Predict"):
    if title and text:
        result = predict_news(title, text)
        st.success(f"The news is predicted as: **{result}**")
    else:
        st.error("Please enter both title and text.")