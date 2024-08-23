from pydantic import BaseModel

class TextInput(BaseModel):
    text: str

class ImageOutput(BaseModel):
    image: str  # This could be a URL or a base64 encoded image string
