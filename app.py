import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.graph_objects as go

from huggingface_hub import hf_hub_download
from utils import Preprocessor

REPO_ID = "amosfang/medical_symptoms_rfc"
FILENAME = "medical_symptoms_rfc.joblib"

# Helper function for feature extraction
def extract_features(X):
    
    vectorizer = joblib.load('vectorizer.pkl') # Load the previously saved vectorizer
    count_matrix = vectorizer.transform(X)  

    tfidf_transformer = joblib.load('tfidf_transformer.pkl')  # Load the saved TF-IDF transformer
    tfidf_matrix = tfidf_transformer.transform(count_matrix) 

    return tfidf_matrix

# Helper function for prediction
def predict_symptoms(symptoms_text):
    
    # Predict disease
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = joblib.load(model_path)

    data_preprocessor = Preprocessor()
    X = data_preprocessor.transform(pd.Series(symptoms_text))
    
    tfidf_matrix = extract_features(X)
    prediction_prob = model.predict_proba(tfidf_matrix)

    # Get top 5 predictions for each sample
    top_n = 5
    predictions = np.argsort(prediction_prob, axis=1)[:, -top_n:][:, ::-1]  # Get indices of top 5 diseases
    top_n_probs = np.sort(prediction_prob, axis=1)[:, -top_n:][:, ::-1]

    return predictions, top_n_probs
    
def run():
    # Streamlit UI
    st.set_page_config(page_title="Disease Predictor", layout="centered")
    
    # Header
    st.title("Disease Predictor")
    st.subheader("Analyze and predict the disease given symptoms")
    
    # User Input
    st.markdown("### Enter Your Symptoms")
    user_symptoms = st.text_area(
        "Type or paste symptoms below to predict disease.",
        placeholder="fever and sore throat",
    )
    
    # Submit Button
    if st.button("Predict Disease"):
        if user_symptoms.strip():
            # Make prediction
            predictions, top_n_probs = predict_symptoms(user_symptoms)
            prediction_diseases = diseases_df.iloc[predictions[0]]["disease_name"]
    
            # Display Results
            st.markdown(f"Disease: {prediction_diseases.tolist()}")
            # st.markdown(f"### Disease: **{' '.join(prediction_diseases.tolist())}**")
            # st.markdown(f"**Confidence:** {' '.joint(top_n_probs.tolist())}")
            
            # Plotly Bar Chart for Probabilities
            # fig = go.Figure(data=[
            #     go.Bar(
            #         x=prediction_diseases,
            #         y=top_n_probs[0],
            #         textposition='auto',
            #     )
            # ])
            # fig.update_layout(
            #     title="Prediction Probabilities",
            #     xaxis_title="Diseases",
            #     yaxis_title="Probability",
            #     template="plotly_white"
            # )
            # st.plotly_chart(fig)
            
            st.info(
                "Disease prediction is based on trained machine learning algorithms using advanced text processing techniques."
            )
        else:
            st.error("Please enter a valid patient symptom before clicking 'Predict Disease'.")
    
    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit | © 2025 Medical Insights AI")

if __name__ == "__main__":
    # Load JSON from a file
    with open("diseases.json", "r", encoding="utf-8") as json_file:
        diseases_dict = json.load(json_file)
        
    diseases_df = pd.DataFrame(diseases_dict).drop_duplicates()
    diseases_df.set_index("id", inplace=True)

    # Run Streamlit App
    run()

