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
import gc
from contextlib import contextmanager
from flask_caching import Cache
import logging
from logging.handlers import RotatingFileHandler
import uuid

# Initialize Flask app with caching
app = Flask(__name__)
cache_config = {
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300
}
app.config.from_mapping(cache_config)
cache = Cache(app)

# Setup logging
if not app.debug:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/ser.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('SER startup')

@contextmanager
def cleanup_memory():
    try:
        yield
    finally:
        gc.collect()
        tf.keras.backend.clear_session()

class ModelManager:
    _instance = None
    _models = None
    _scaler = None
    _label_encoder = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_models(self):
        if self._models is None:
            self._models, self._scaler, self._label_encoder = self._load_models()
        return self._models, self._scaler, self._label_encoder

    def _load_models(self):
        ensemble_models = []
        ensemble_paths = ["multi_model5a.keras"]
        bucket_name = "ser_models"
        model_folder = "models"
        scaler_folder = "scalers"
        destination_folder = "/tmp"

        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-credentials.json'
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Download and load models
        for path in ensemble_paths:
            source_blob_name = f"{model_folder}/{path}"
            destination_file_name = f"{destination_folder}/{path}"
            blob = bucket.blob(source_blob_name)
            blob.download_to_filename(destination_file_name)
            model = load_model(destination_file_name)
            ensemble_models.append(model)

        # Download and load scaler
        scaler_path = "scaler_multi.joblib"
        source_blob_name = f"{scaler_folder}/{scaler_path}"
        destination_file_name = f"{destination_folder}/{scaler_path}"
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        scaler = joblib.load(destination_file_name)

        # Download and load label encoder
        label_encoder_path = "label_multi.joblib"
        source_blob_name = f"{scaler_folder}/{label_encoder_path}"
        destination_file_name = f"{destination_folder}/{label_encoder_path}"
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)
        label_encoder = joblib.load(destination_file_name)

        return ensemble_models, scaler, label_encoder

@cache.memoize(timeout=300)
def frft(x, alpha):
    N = len(x)
    t = np.arange(N)
    kernel = np.exp(-1j * np.pi * alpha * t**2 / N)
    return scipy.signal.fftconvolve(x, kernel, mode='same')

@cache.memoize(timeout=300)
def extract_features(data, sample_rate):
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

def predict_emotion(audio_file, scaler, window_size=3.0, hop_size=1.0):
    with cleanup_memory():
        data, sr = librosa.load(audio_file, sr=16000, duration=30)
        data = trim_silences(data, sr)
        data = normalize_audio(data)
        windows = generate_windows(data, window_size, hop_size, sr)
        
        if len(windows) == 0:
            return {label: "0.00%" for label in ModelManager.get_instance().get_models()[2].classes_}
            
        batch_size = 32
        emotion_probs = np.zeros(len(ModelManager.get_instance().get_models()[2].classes_))
        
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i + batch_size]
            batch_features = []
            
            for window in batch:
                features = extract_features(window, sr)
                features_scaled = scaler.transform(features.reshape(1, -1))
                batch_features.append(features_scaled.reshape(features_scaled.shape[1], 1))
            
            batch_features = np.array(batch_features)
            models = ModelManager.get_instance().get_models()[0]
            batch_probs = np.mean([model.predict(batch_features, verbose=0) 
                                 for model in models], axis=0)
            emotion_probs += np.sum(batch_probs, axis=0)
        
        emotion_probs /= len(windows)
        return {label: f"{prob * 100:.2f}%" 
                for label, prob in zip(ModelManager.get_instance().get_models()[2].classes_, emotion_probs)}

# Your existing helper functions remain the same
def normalize_audio(audio):
    original_max = np.abs(audio).max()
    audio = audio.astype(np.float32)
    normalized_audio = np.clip(audio / original_max, -1.0, 1.0)
    return normalized_audio

def trim_silences(data, sr, top_db=35):
    trimmed_data, _ = librosa.effects.trim(data, top_db=top_db)
    return trimmed_data

def generate_windows(data, window_size, hop_size, sr):
    num_samples = len(data)
    window_samples = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    windows = []
    for i in range(0, num_samples - window_samples + 1, hop_samples):
        window = data[i:i + window_samples]
        windows.append(window)
    return windows

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    try:
        audio_file = request.files['audio']
        temp_path = os.path.join('/tmp', f"{uuid.uuid4()}.wav")
        
        with open(temp_path, 'wb') as f:
            audio_file.save(f)
        
        model_manager = ModelManager.get_instance()
        models, scaler, label_encoder = model_manager.get_models()
        
        with cleanup_memory():
            predictions, transcription = process_audio_file(temp_path)
        
        os.remove(temp_path)
        
        return jsonify({
            "Emotion Probabilities": predictions,
            "Transcription": transcription
        })
        
    except Exception as e:
        app.logger.error(f'Error processing audio: {str(e)}')
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File is too large"}), 413

@app.errorhandler(500)
def internal_error(error):
    app.logger.error('Server Error: %s', str(error))
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=False)
