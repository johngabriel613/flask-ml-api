from flask import jsonify
import numpy as np
import librosa
from server.models.load_all_model import models

def test_exp4_model(audio_path):
    model = models['exp4']
    classes=['artifact', 'murmur', 'normal', 'extrastole', 'extrahls']
    # Load and preprocess the audio file
    duration = 10  # as per original configuration
    sr = 22050    # sampling rate
    input_length = sr * duration

    # Load audio file
    audio_data, _ = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Fix the audio length to match the model's input requirements
    if len(audio_data) < input_length:
        audio_data = np.pad(audio_data, (0, max(0, input_length - len(audio_data))), 'constant')
    else:
        audio_data = audio_data[:input_length]
    
    # Extract MFCC features
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=52).T, axis=0)
    mfccs = mfccs.reshape(1, -1, 1)  # Reshape for model input

    # Predict using the model
    predictions = model.predict(mfccs)
    predicted_class_idx = np.argmax(predictions)
    confidence_score = predictions[0][predicted_class_idx]

    # Get class label and confidence
    predicted_class = classes[predicted_class_idx]
    
    return jsonify({
            'classification': predicted_class,
            'prediction_score': float(confidence_score)
        })
