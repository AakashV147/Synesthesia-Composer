import base64
import numpy as np
import os
import tensorflow as tf
import json


def get_file_path(file_name: str, directory: str) -> str:
    """
    Construct the file path for the given file name within the specified directory.

    Parameters:
    file_name (str): The name of the file.
    directory (str): The directory where the file is located.

    Returns:
    str: The full file path.
    """
    return os.path.join(directory, file_name)

def encode_image_to_base64(image: np.ndarray) -> str:
    """Encodes a numpy array image to a base64 string."""
    image_pil = Image.fromarray(image)
    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """Decodes a base64 string to a numpy array image."""
    image_data = base64.b64decode(base64_str)
    image_pil = Image.open(io.BytesIO(image_data))
    return np.array(image_pil)

def convert_to_wav_format(sound_data: np.ndarray) -> bytes:
    """Converts numpy array sound data to WAV format bytes."""
    # Placeholder for actual WAV file creation from numpy array
    pass

def save_file(data, path):
    # Function implementation
    pass

def save_file(content, file_name, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_path = os.path.join(save_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(content)
    
    return file_path

def preprocess_music_input(music_input):
    """
    Preprocess the music input to extract features suitable for the model.
    This function assumes that `music_input` is a path to a JSON file or raw audio data.
    Adjust this function based on your data format and feature extraction method.
    """
    # Load raw audio data (assuming `music_input` is a path to a JSON file with features)
    with open(music_input, 'r') as f:
        data = json.load(f)
    
    # Example of feature extraction (replace with your actual preprocessing)
    features = np.array(data['features'])  # Assuming 'features' is a key in your JSON
    
    # Normalize or preprocess features if needed
    preprocessed_features = tf.convert_to_tensor(features, dtype=tf.float32)

    # Add batch dimension
    preprocessed_features = tf.expand_dims(preprocessed_features, axis=0)

    return preprocessed_features
