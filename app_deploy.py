import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import io

# Load trained model and label encoder
model = tf.keras.models.load_model('emotion_cnn_model.h5')

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

st.set_page_config(page_title="Emotion Classifier", layout="centered")
st.title("🎧 Speech Emotion Recognition")
st.markdown("Upload an audio file (WAV format), and the model will predict the emotion.")

uploaded_file = st.file_uploader("Choose a .wav file", type="wav")

def extract_mfcc(file_path, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded file to a buffer
    bytes_data = uploaded_file.read()
    audio_buffer = io.BytesIO(bytes_data)

    # Extract MFCC
    mfcc = extract_mfcc(audio_buffer)

    # Reshape to match model input
    mfcc_reshaped = mfcc.reshape(1, -1, 1)  # (1, 40, 1)

    # Predict
    prediction = model.predict(mfcc_reshaped)
    predicted_class = np.argmax(prediction)
    predicted_label = le.inverse_transform([predicted_class])[0]

    st.success(f"🗣️ Predicted Emotion: **{predicted_label.capitalize()}**")

 