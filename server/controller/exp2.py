from flask import jsonify
import numpy as np
import librosa
from server.models.load_all_model import models

# Define classes and create mappings
classes = ['artifact', 'extrahls', 'extrastole', 'murmur', 'normal']
index_to_class = {i: label for i, label in enumerate(classes)}

# Function to extract and preprocess the spectrogram
def extract_spectrogram(file_path, n_mels=128, max_pad_len=128):
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)

        # Generate Mel spectrogram and convert to log scale (dB)
        spectrogram = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels), ref=np.max)
        # Normalize and pad/truncate to max_pad_len
        spectrogram = (spectrogram - np.min(spectrogram)) / (np.max(spectrogram) - np.min(spectrogram))
        return np.pad(spectrogram, ((0, 0), (0, max(0, max_pad_len - spectrogram.shape[1]))), mode='constant')[:, :max_pad_len]
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to load model and make predictions
def test_exp2_model(audio_file_path):
    model = models['exp2']
    spectrogram = extract_spectrogram(audio_file_path)
    if spectrogram is None:
        return None, None
    
    # Expand dimensions and make predictions
    predictions = model.predict(spectrogram[np.newaxis, ..., np.newaxis])
    confidence_score = float(np.max(predictions))  # Ensure JSON-serializable
    predicted_class = index_to_class[np.argmax(predictions)]
    
    return jsonify({
        "prediction_score": confidence_score,
        "classification": predicted_class
    })