from fastapi import APIRouter
from pydantic import BaseModel

# Import the services for each modality
from app.services.text_service import generate_visual_from_text
from app.services.image_service import generate_sound_from_image
from app.services.music_service import generate_visual_from_music

api_router = APIRouter()

# Schemas for input data
class TextInput(BaseModel):
    text: str

class ImageInput(BaseModel):
    image_url: str

class MusicInput(BaseModel):
    music_url: str

# Text to Visual
@api_router.post("/text-to-visual/")
async def text_to_visual(input: TextInput):
    visual = generate_visual_from_text(input.text)
    return {"visual_output": visual}

# Image to Sound
@api_router.post("/image-to-sound/")
async def image_to_sound(input: ImageInput):
    sound = generate_sound_from_image(input.image_url)
    return {"sound_output": sound}

# Music to Visual
@api_router.post("/music-to-visual/")
async def music_to_visual(input: MusicInput):
    visual = generate_visual_from_music(input.music_url)
    return {"visual_output": visual}
