from transformers import pipeline

def load_text_to_image_model():
    return pipeline("text-to-image-generation", model="CompVis/stable-diffusion-v1-4")

class TextToImageModel:
    def __init__(self):
        self.model = load_text_to_image_model()

    def generate_image(self, text: str):
        return self.model(text)
