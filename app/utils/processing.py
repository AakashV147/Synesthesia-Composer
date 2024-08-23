import base64
import numpy as np

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
