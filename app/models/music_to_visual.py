import os
import logging
import tensorflow as tf
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_tfrecord_files(directory):
    """Retrieve all TFRecord files from the specified directory."""
    tfrecord_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.tfrecord')]
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in directory: {directory}")
    return tfrecord_files

def get_youtube8m_feature_shape(tfrecord_files):
    """Extract feature shape from the TFRecord files."""
    logger.info(f"Extracting feature shape from: {tfrecord_files}")

    raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
    
    feature_description = {
        'audio_embedding': tf.io.FixedLenFeature([10, 128], tf.string),  # Update based on your actual data
    }

    def _parse_function(proto):
        parsed_features = tf.io.parse_single_example(proto, feature_description)
        return parsed_features['audio_embedding']

    parsed_dataset = raw_dataset.map(_parse_function)
    
    for feature in parsed_dataset.take(1):  # Get the first record's feature shape
        audio_embedding = tf.io.decode_raw(feature.numpy(), tf.float32)
        shape = audio_embedding.shape
        logger.info(f"Extracted feature shape: {shape}")
        return shape

def build_music_to_visual_model(input_shape):
    """Build and compile the music-to-visual model."""
    logger.info(f"Building model with input shape: {input_shape}")
    
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(256, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(3, activation='sigmoid')  # Adjust based on output dimensions
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    logger.info("Model built successfully.")
    return model

class TrainingLogger(tf.keras.callbacks.Callback):
    """Custom callback to log training progress."""
    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"Epoch {epoch+1}/{self.params['epochs']}, Loss: {logs['loss']:.4f}")

def train_music_to_visual_model(model, train_dataset, val_dataset, epochs=10):
    """Train the music-to-visual model."""
    logger.info("Starting model training.")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[TrainingLogger()],
        verbose=0
    )
    logger.info("Model training completed.")
    return history

def main():
    train_directory = "/Users/aakashvenkatraman/Documents/GitHub/Synesthesia-Composer/data/inputs/music_to_visual/train/Validate"  # Update with actual path
    val_directory = "/Users/aakashvenkatraman/Documents/GitHub/Synesthesia-Composer/data/inputs/music_to_visual/test/test"  # Update with actual path
    
    train_file_paths = get_tfrecord_files(train_directory)
    val_file_paths = get_tfrecord_files(val_directory)
    
    logger.info("Extracting input shape.")
    input_shape = get_youtube8m_feature_shape(train_file_paths)
    logger.info(f"Extracted input shape: {input_shape}")
    
    logger.info("Building model.")
    model = build_music_to_visual_model(input_shape)
    model.summary()

    logger.info("Preparing datasets.")
    def _parse_function(proto):
        feature_description = {
            'audio_embedding': tf.io.FixedLenFeature([10, 128], tf.string),
            'visual_output': tf.io.FixedLenFeature([256, 256, 3], tf.string),  # Update based on your actual data
        }
        parsed_features = tf.io.parse_single_example(proto, feature_description)
        audio_embedding = tf.io.decode_raw(parsed_features['audio_embedding'], tf.float32)
        visual_output = tf.io.decode_raw(parsed_features['visual_output'], tf.uint8)
        return audio_embedding, visual_output

    train_dataset = tf.data.TFRecordDataset(train_file_paths).map(_parse_function).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.TFRecordDataset(val_file_paths).map(_parse_function).batch(32).prefetch(tf.data.AUTOTUNE)

    train_music_to_visual_model(model, train_dataset, val_dataset)
    
    model.save("music_to_visual.h5")
    logger.info("Model saved.")

if __name__ == "__main__":
    main()
