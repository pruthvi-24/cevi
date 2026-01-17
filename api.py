from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from pipeline import analyze_food_image

app = FastAPI(title="Food Water Footprint API")

# ✅ ADD THIS (for HTML / browser access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow browser
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OPTIONAL but useful (check API running)
@app.get("/")
def root():
    return {"status": "API is running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = analyze_food_image(image)

    return {
        "status": "success",
        "data": result
    }
