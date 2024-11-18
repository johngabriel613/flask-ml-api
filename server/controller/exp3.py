from flask import jsonify
import numpy as np
import librosa
from server.models.load_all_model import models



# Function to preprocess audio and get prediction
def test_exp3_model(audio_path, input_shape=(128, 128)):
    model = models['exp3']

    classes = ['artifact', 'extrahls', 'extrastole', 'murmur', 'normal']
    # Load the audio file
    audio, sample_rate = librosa.load(audio_path, sr=None)
    
    # Generate Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=input_shape[0])
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Normalize spectrogram
    log_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / (np.max(log_spectrogram) - np.min(log_spectrogram))
    
    # Pad or truncate the spectrogram to match the input shape
    if log_spectrogram.shape[1] < input_shape[1]:
        pad_width = input_shape[1] - log_spectrogram.shape[1]
        log_spectrogram = np.pad(log_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        log_spectrogram = log_spectrogram[:, :input_shape[1]]
    
    # Reshape for model input
    log_spectrogram = log_spectrogram.reshape(1, input_shape[0], input_shape[1], 1)
    
    # Make prediction
    predictions = model.predict(log_spectrogram)
    
    # Get the predicted class and confidence
    predicted_index = np.argmax(predictions)
    predicted_class = classes[predicted_index]
    confidence_score = float(predictions[0][predicted_index])  # Convert to float for JSON compatibility
    
    # Return JSON response
    return jsonify({
        "prediction_score": confidence_score,
        "classification": predicted_class
    })





