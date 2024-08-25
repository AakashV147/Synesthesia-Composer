import os
import tensorflow as tf
from fastapi import HTTPException
from app.models.music_to_visual import build_music_to_visual_model, train_music_to_visual_model
from app.utils.processing import preprocess_music_input

# Path where the trained model is saved
MODEL_PATH = "/Users/aakashvenkatraman/Documents/GitHub/Synesthesia-Composer/models/huggingface/saved_models/music_to_visual.h5"

# Load the pre-trained music-to-visual model
def load_music_to_visual_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=500, detail="Model not found. Please train the model first.")
    
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Generate visuals from the music input
def generate_visual_from_music(music_input):
    model = load_music_to_visual_model()

    # Preprocess the actual music input
    preprocessed_music_input = preprocess_music_input(music_input)  # Pass the actual music input

    # Make predictions
    predictions = model.predict(preprocessed_music_input)

    # Convert predictions to a visual output (for example, labels or image IDs)
    visuals = predictions.argmax(axis=-1)  # Example: return class labels as visuals
    return visuals.tolist()

# If you need a training method
def train_and_save_music_to_visual_model():
    model = train_music_to_visual_model()
    model.save(MODEL_PATH)
