import numpy as np
from PIL import Image
from ..utils.processing import save_file, get_file_path

def generate_sound_from_image(image_url: str):
    image_path = get_file_path(image_url, "data/inputs/images/")
    image = Image.open(image_path)

    # Perform image-to-sound conversion (actual logic will replace this placeholder)
    sound_data = np.random.rand(44100)

    # Save generated sound file
    sound_file_name = image_url.split('.')[0] + ".wav"
    sound_file_path = save_file(sound_data, sound_file_name, "data/outputs/sounds/")

    return sound_file_path
