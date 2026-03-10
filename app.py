import streamlit as st
import pickle

st.set_page_config(page_title="Disease Prediction App", page_icon="🩺", layout="centered")

st.title("🩺 Disease Prediction from Symptoms")
st.write("Enter symptoms separated by commas to predict the disease.")

# Load saved files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# User input
symptoms = st.text_area("Enter Symptoms", placeholder="Example: fever, headache, nausea")

# Prediction button
if st.button("Predict Disease"):

    if symptoms.strip() == "":
        st.warning("Please enter symptoms.")
    
    else:
        # Convert input to list
        text = [symptoms]

        # Transform text
        text_vector = vectorizer.transform(text)

        # Predict
        prediction = model.predict(text_vector)

        # Decode disease
        disease = encoder.inverse_transform(prediction)

        # Show result
        st.success(f"Predicted Disease: {disease[0]}")