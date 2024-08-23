from fastapi import UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from .utils.processing import save_file
from .services.text_service import generate_visual_from_text
from .services.image_service import generate_sound_from_image
from .services.music_service import generate_visual_from_music

@api_router.post("/upload-text/")
async def upload_text(text: str):
    file_name = f"text_{hash(text)}.txt"
    file_path = save_file(text.encode(), file_name, "data/inputs/text/")
    return {"file_name": file_name, "file_path": file_path}

@api_router.post("/upload-image/")
async def upload_image(image: UploadFile = File(...)):
    file_name = image.filename
    file_content = await image.read()
    file_path = save_file(file_content, file_name, "data/inputs/images/")
    return {"file_name": file_name, "file_path": file_path}

@api_router.post("/upload-music/")
async def upload_music(music: UploadFile = File(...)):
    file_name = music.filename
    file_content = await music.read()
    file_path = save_file(file_content, file_name, "data/inputs/music/")
    return {"file_name": file_name, "file_path": file_path}

@api_router.get("/download-output/")
async def download_output(file_name: str):
    try:
        file_path = get_file_path(file_name, "data/outputs/")
        return FileResponse(file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
