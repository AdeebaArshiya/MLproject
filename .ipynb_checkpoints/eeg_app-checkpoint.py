import streamlit as st
import joblib
import numpy as np

# Load model and label encoder
model = joblib.load('eeg_model.pkl')
le = joblib.load('label_encoder.pkl')

# Use actual scaler means and stds
mean_alpha = 0.00500386
mean_beta = 0.0055358
mean_gamma = 0.00651647

std_alpha = 1.0210069
std_beta = 1.04218306
std_gamma = 1.02454405

st.title("EEG Emotion Predictor")
st.write("Enter raw Alpha, Beta, Gamma EEG feature values:")

alpha_raw = st.number_input("Alpha (raw)", value=float(mean_alpha), format="%.6f")
beta_raw = st.number_input("Beta (raw)", value=float(mean_beta), format="%.6f")
gamma_raw = st.number_input("Gamma (raw)", value=float(mean_gamma), format="%.6f")

if st.button("Predict"):
    alpha_scaled = (alpha_raw - mean_alpha) / std_alpha
    beta_scaled  = (beta_raw - mean_beta) / std_beta
    gamma_scaled = (gamma_raw - mean_gamma) / std_gamma

    input_scaled = np.array([[alpha_scaled, beta_scaled, gamma_scaled]])
    prediction = model.predict(input_scaled)
    emotion = le.inverse_transform(prediction)[0]
    st.success(f"Predicted Emotion: {emotion}")
