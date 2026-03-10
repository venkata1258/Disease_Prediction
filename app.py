import streamlit as st
import pickle
import re

# Page configuration
st.set_page_config(page_title="Medical Report Analyzer", page_icon="🩺", layout="centered")

st.title("🩺 Medical Report Analyzer")
st.write("Enter symptoms separated by commas to predict the possible disease.")

# Load models safely
@st.cache_resource
def load_models():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)

        with open("label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)

        return model, vectorizer, encoder

    except Exception as e:
        st.error("Model files could not be loaded. Please check deployment files.")
        st.stop()

# Load models
model, vectorizer, encoder = load_models()

# User input
symptoms = st.text_area(
    "Enter Symptoms",
    placeholder="Example: anxiety and nervousness, shortness of breath, insomnia"
)

# Clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z, ]', '', text)
    return text

# Prediction
if st.button("Predict Disease"):

    if symptoms.strip() == "":
        st.warning("Please enter symptoms.")

    else:
        cleaned_input = clean_text(symptoms)

        text = [cleaned_input]

        text_vector = vectorizer.transform(text)

        prediction = model.predict(text_vector)

        disease = encoder.inverse_transform(prediction)

        st.success(f"Predicted Disease: {disease[0]}")

st.markdown("---")
st.caption("AI Medical Report Analyzer | Machine Learning Project")