from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from .utils.processing import save_file, get_file_path  # Ensure get_file_path is implemented

# Create an APIRouter instance
api_router = APIRouter()

# Define routes using the APIRouter instance
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

# Create the FastAPI app and include the router
app = FastAPI()
app.include_router(api_router)

app = FastAPI()

@app.post("/train-music-to-visual/")
async def train_music_to_visual():
    try:
        train_and_save_music_to_visual_model()
        return {"detail": "Model trained and saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-visual/")
async def generate_visual(music_input: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"temp_{music_input.filename}"
        with open(file_path, "wb") as f:
            f.write(await music_input.read())
        
        # Generate visual output
        visuals = generate_visual_from_music(file_path)
        
        # Clean up temporary file
        os.remove(file_path)
        
        return {"visuals": visuals}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
