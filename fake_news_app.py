import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model, vectorizer = load_model()

st.title("Fake News Detector")
st.markdown("Enter a news title and article text to predict if it's **Real** or **Fake**.")

st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Enter a news title in the first input field.
2. Enter the news article text in the second field.
3. Click the **Predict** button to see the result.
4. Note: This model is trained on a specific dataset and may not be 100% accurate.
""")

st.sidebar.header("Example Input")
st.sidebar.markdown("""
- **Title**: Economy Grows by 3% This Quarter
- **Text**: Government reports show a steady increase in GDP...
""")

title = st.text_input("News Title", placeholder="Enter the news headline...")
text = st.text_area("News Text", placeholder="Enter the news article text...", height=200)

def predict_news(title, text):
    combined_text = title + " " + text
    text_transformed = vectorizer.transform([combined_text])
    prediction = model.predict(text_transformed)
    label = "Real" if prediction[0] == 1 else "Fake"
    # Check if model supports predict_proba (SVC does, LinearSVC doesn't)
    confidence = None
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(text_transformed)[0]
        confidence = probability[1] if prediction[0] == 1 else probability[0]
    return label, confidence

if st.button("Predict"):
    if title.strip() and text.strip():
        result, confidence = predict_news(title, text)
        if confidence is not None:
            st.success(f"The news is predicted as: **{result}** (Confidence: {confidence:.2%})")
        else:
            st.success(f"The news is predicted as: **{result}**")
    else:
        st.error("Please enter both a title and text to analyze.")

st.markdown("---")
st.markdown("Built with Streamlit and Scikit-learn. Model trained on fake and real news datasets.")
