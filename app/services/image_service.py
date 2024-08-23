from ..models.image_to_sound import image_to_sound_model

def generate_sound_from_image(image_url: str) -> bytes:
    image_path = get_file_path(image_url, "data/inputs/images/")
    image = Image.open(image_path)

    # Generate sound from the image
    sound_wav = image_to_sound_model(np.array(image))
    
    # Save sound as a WAV file
    sound_file_name = image_url.split('.')[0] + ".wav"
    save_file(sound_wav, sound_file_name, "data/outputs/sounds/")

    return sound_file_name
