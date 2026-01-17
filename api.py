from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

from pipeline import analyze_food_image

app = FastAPI(title="Food Water Footprint API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (if any)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html explicitly
@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

# API endpoint (THIS WILL NOW WORK)
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return analyze_food_image(image)
