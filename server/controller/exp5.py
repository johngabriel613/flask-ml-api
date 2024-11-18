from flask import jsonify
import numpy as np
import librosa
from server.models.load_all_model import models

# Define the prediction function
def test_exp5_model(file_path):
    try:
        model = models['exp5']
        # Load and preprocess the audio file
        SAMPLE_RATE = 22050
        MAX_SOUND_CLIP_DURATION = 10
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_SOUND_CLIP_DURATION)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=52).T, axis=0)
        feature = np.array(mfccs).reshape([-1, 1])
        input_data = np.expand_dims(feature, axis=0)

        # Make prediction
        prediction = model.predict(input_data)

        # Get the predicted class and score
        predicted_class_index = np.argmax(prediction)
        prediction_score = float(np.max(prediction))

        # Map predicted class integer to label
        int_to_label = {0: 'artifact', 1: 'murmur', 2: 'normal', 3: 'extrastole', 4: 'extrahls'}
        predicted_class = int_to_label[predicted_class_index]

        return jsonify({
            'classification': predicted_class,
            'prediction_score': prediction_score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
