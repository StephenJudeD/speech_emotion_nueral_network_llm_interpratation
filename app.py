from flask import Flask, request, jsonify, render_template
import os
import joblib
import numpy as np
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import scipy.signal
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.cloud import storage
import requests
import openai
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
ENSEMBLE_MODELS = None
SCALER = None
LABEL_ENCODER = None

def download_model(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        logger.info(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def initialize_models():
    """Initialize models on startup"""
    global ENSEMBLE_MODELS, SCALER, LABEL_ENCODER
    
    try:
        ensemble_models = []
        ensemble_paths = ["multi_model5a.keras"]
        
        bucket_name = "ser_models"
        model_folder = "models"
        scaler_folder = "scalers"
        destination_folder = "/tmp"

        # Set Google credentials
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-credentials.json'

        # Download and load models
        for path in ensemble_paths:
            source_blob_name = f"{model_folder}/{path}"
            destination_file_name = f"{destination_folder}/{path}"
            download_model(bucket_name, source_blob_name, destination_file_name)
            model = load_model(destination_file_name)
            ensemble_models.append(model)

        # Download and load scaler
        scaler_path = "scaler_multi.joblib"
        source_blob_name = f"{scaler_folder}/{scaler_path}"
        destination_file_name = f"{destination_folder}/{scaler_path}"
        download_model(bucket_name, source_blob_name, destination_file_name)
        scaler = joblib.load(destination_file_name)

        # Download and load label encoder
        label_encoder_path = "label_multi.joblib"
        source_blob_name = f"{scaler_folder}/{label_encoder_path}"
        destination_file_name = f"{destination_folder}/{label_encoder_path}"
        download_model(bucket_name, source_blob_name, destination_file_name)
        label_encoder = joblib.load(destination_file_name)

        ENSEMBLE_MODELS = ensemble_models
        SCALER = scaler
        LABEL_ENCODER = label_encoder
        
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        raise

def frft(x, alpha):
    """Fractional Fourier Transform implementation"""
    N = len(x)
    t = np.arange(N)
    kernel = np.exp(-1j * np.pi * alpha * t**2 / N)
    return scipy.signal.fftconvolve(x, kernel, mode='same')

def extract_features(data, sample_rate):
    """Extract audio features"""
    try:
        mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=13).T, axis=0)
        delta_mfcc = np.mean(librosa.feature.delta(mfcc).T, axis=0)
        acceleration_mfcc = np.mean(librosa.feature.delta(mfcc, order=2).T, axis=0)
        mel_spectrogram = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)

        alpha_values = np.linspace(0.1, 0.9, 9)
        frft_features = np.array([])

        for alpha in alpha_values:
            frft_result = frft(data, alpha)
            frft_features = np.hstack((frft_features, np.mean(frft_result.real, axis=0)))

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)

        return np.hstack((mfcc, delta_mfcc, acceleration_mfcc, mel_spectrogram, frft_features, spectral_centroid))
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def normalize_audio(audio):
    """Normalize audio data"""
    try:
        original_max = np.abs(audio).max()
        audio = audio.astype(np.float32)
        normalized_audio = np.clip(audio / original_max, -1.0, 1.0)
        return normalized_audio
    except Exception as e:
        logger.error(f"Error normalizing audio: {str(e)}")
        raise

def trim_silences(data, sr, top_db=35):
    """Remove silence from audio"""
    try:
        trimmed_data, _ = librosa.effects.trim(data, top_db=top_db)
        return trimmed_data
    except Exception as e:
        logger.error(f"Error trimming silences: {str(e)}")
        raise

def generate_windows(data, window_size, hop_size, sr):
    """Generate sliding windows from audio data"""
    try:
        num_samples = len(data)
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)

        windows = []
        for i in range(0, num_samples - window_samples + 1, hop_samples):
            window = data[i:i + window_samples]
            windows.append(window)

        return windows
    except Exception as e:
        logger.error(f"Error generating windows: {str(e)}")
        raise

def predict_emotion(audio_file, scaler, window_size=3.0, hop_size=1.0):
    """Predict emotions from audio file"""
    try:
        data, sr = librosa.load(audio_file, sr=16000)
        data = trim_silences(data, sr)
        data = normalize_audio(data)

        windows = generate_windows(data, window_size, hop_size, sr)

        if len(windows) == 0:
            return {label: "0.00%" for label in LABEL_ENCODER.classes_}

        emotion_probs = np.zeros(len(LABEL_ENCODER.classes_))

        for window in windows:
            features = extract_features(window, sr)
            features_scaled = scaler.transform(features.reshape(1, -1))
            features_reshaped = features_scaled.reshape(1, features_scaled.shape[1], 1)

            window_probs = np.mean([model.predict(features_reshaped, verbose=0)[0] 
                                  for model in ENSEMBLE_MODELS], axis=0)
            emotion_probs += window_probs

        emotion_probs /= len(windows)
        
        emotion_probability_distribution = {
            label: f"{prob * 100:.2f}%" 
            for label, prob in zip(LABEL_ENCODER.classes_, emotion_probs)
        }

        return emotion_probability_distribution
    except Exception as e:
        logger.error(f"Error predicting emotion: {str(e)}")
        raise

def transcribe_audio(audio_file_path):
    """Transcribe audio to text"""
    try:
        audio = AudioSegment.from_file(audio_file_path)
        wav_file_path = "/tmp/uploaded_audio.wav"
        audio.export(wav_file_path, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file_path) as source:
            audio_data = recognizer.record(source)
            try:
                transcription = recognizer.recognize_google(audio_data)
                return transcription
            except sr.UnknownValueError:
                return "[unrecognized]"
            except sr.RequestError as e:
                return f"Transcription error: {e}"
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise
    finally:
        if os.path.exists(wav_file_path):
            os.remove(wav_file_path)

def get_llm_interpretation(emotional_results, transcription):
    """Get LLM interpretation of emotions"""
    try:
        openai.api_key = os.getenv('OPENAI_API_KEY')
        prompt = f"""
        You are an expert in audio emotion recognition and analysis. Given the following information:

            Audio data details:
            - Emotional recognition results: {emotional_results}
            - Transcript: {transcription}

            Provide a comprehensive interpretation of the emotional content, considering both the
            emotion recognition results and the transcript.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}"
        }

        payload = {
            "model": "gpt-4-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }

        response = requests.post(
            "https://api.openai.com/v1/chat/completions", 
            headers=headers, 
            json=payload
        )
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Error getting LLM interpretation: {str(e)}")
        return f"LLM interpretation error: {str(e)}"

def process_audio_file(audio_file):
    """Process audio file and return results"""
    try:
        prediction = predict_emotion(audio_file, SCALER)
        transcription = transcribe_audio(audio_file)
        llm_interpretation = get_llm_interpretation(prediction, transcription)
        return prediction, transcription, llm_interpretation
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        raise

@app.route('/')
def index():
    """Render index page"""
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded audio file"""
    temp_file_path = None
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files['audio']
        temp_file_path = os.path.join('/tmp', audio_file.filename)
        audio_file.save(temp_file_path)
        logger.info(f"Audio file saved at: {temp_file_path}")

        predictions, transcription, llm_interpretation = process_audio_file(temp_file_path)
        
        response = {
            "Emotion Probabilities": predictions,
            "Transcription": transcription,
            "LLM Interpretation": llm_interpretation,
        }
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == '__main__':
    # Initialize models before starting the app
    initialize_models()
    app.run(debug=True)
