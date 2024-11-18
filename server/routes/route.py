from flask import Blueprint, request, jsonify
from server.controller.exp1 import test_exp1_model
from server.controller.exp2 import test_exp2_model
from server.controller.exp3 import test_exp3_model
from server.controller.exp4 import test_exp4_model
from server.controller.exp5 import test_exp5_model
from server.controller.visualization import get_audio_visualizations


prediction = Blueprint('predict', __name__)

@prediction.route('/exp1', methods=['POST'])
def exp1():
  # Get file from request
    file = request.files["audio_file"]

    # Save the uploaded file to a temporary location
    temp_file_path = "temp_audio_file.wav"
    file.save(temp_file_path)
    
    # Call the make_prediction function with the file path
    prediction = test_exp1_model(temp_file_path)
    
    # Return the prediction response
    return prediction

@prediction.route('/exp2', methods=['POST'])
def exp2():
  # Get file from request
    file = request.files["audio_file"]

    # Save the uploaded file to a temporary location
    temp_file_path = "temp_audio_file.wav"
    file.save(temp_file_path)
    
    # Call the make_prediction function with the file path
    prediction = test_exp2_model(temp_file_path)
    
    # Return the prediction response
    return prediction

@prediction.route('/exp3', methods=['POST'])
def exp3():
  # Get file from request
    file = request.files["audio_file"]

    # Save the uploaded file to a temporary location
    temp_file_path = "temp_audio_file.wav"
    file.save(temp_file_path)
    
    # Call the make_prediction function with the file path
    prediction = test_exp3_model(temp_file_path)
    
    # Return the prediction response
    return prediction

@prediction.route('/exp4', methods=['POST'])
def exp4():
  # Get file from request
    file = request.files["audio_file"]

    # Save the uploaded file to a temporary location
    temp_file_path = "temp_audio_file.wav"
    file.save(temp_file_path)
    
    # Call the make_prediction function with the file path
    prediction = test_exp4_model(temp_file_path)
    
    # Return the prediction response
    return prediction

@prediction.route('/exp5', methods=['POST'])
def exp5():
  # Get file from request
    file = request.files["audio_file"]

    # Save the uploaded file to a temporary location
    temp_file_path = "temp_audio_file.wav"
    file.save(temp_file_path)
    
    # Call the make_prediction function with the file path
    prediction = test_exp5_model(temp_file_path)
    
    # Return the prediction response
    return prediction

@prediction.route('/visualization', methods=['POST'])
def visualization():
    file = request.files["audio_file"]

    # Save the uploaded file to a temporary location
    temp_file_path = "temp_audio_file.wav"
    file.save(temp_file_path)
    images = get_audio_visualizations(temp_file_path)
    return jsonify(images)


@prediction.route('/test', methods=['GET'])
def test():
    return jsonify({"msg": "hello"})