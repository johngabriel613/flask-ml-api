from flask import jsonify
import librosa
import numpy as np
from server.models.load_all_model import models


def test_exp1_model(audio_file):
    # Load the saved model
    model = models['exp1']

    # Set the threshold
    threshold = 0.5
    
    def preprocess_audio(file_path, n_mels=128, max_pad_len=128):
        try:
            audio, sample_rate = librosa.load(file_path, sr=None)
            
            # Generate Mel spectrogram
            spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=n_mels)
            log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            log_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / (np.max(log_spectrogram) - np.min(log_spectrogram))
            
            # Pad or truncate the spectrogram to a fixed length
            if log_spectrogram.shape[1] < max_pad_len:
                pad_width = max_pad_len - log_spectrogram.shape[1]
                log_spectrogram = np.pad(log_spectrogram, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                log_spectrogram = log_spectrogram[:, :max_pad_len]
                
            return log_spectrogram
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    # Extract features from the provided audio file
    spectrogram = preprocess_audio(audio_file)
    
    if spectrogram is None:
        print("Failed to process the audio file.")
        return jsonify({"error": "Failed to process the audio file"}), 400
    
    # Reshape the spectrogram for model input
    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # Model expects a batch dimension
    
    # Get prediction
    prediction_score = model.predict(spectrogram)[0][0]  # Single output neuron for binary classification
    
    # Classify based on threshold
    classification = "normal" if prediction_score >= threshold else "abnormal"
    
    # Return only prediction score and classification as JSON
    return jsonify({
        "prediction_score": float(prediction_score),
        "classification": classification
    })
