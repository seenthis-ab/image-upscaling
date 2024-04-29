from fastapi import FastAPI, HTTPException, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from google.cloud import storage
from .image_upscaling import upscale
import numpy as np
import cv2
import uuid
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO)


app = FastAPI()

# Make a pydantic model for the request body
class ImageRequest(BaseModel):
    image_url: str

# Initialize Google Cloud Storage client
storage_client = storage.Client()
bucket_name = "image-upscaling"
bucket = storage_client.bucket(bucket_name)

def download_image(image_url: str) -> Image:
    # Download the image file
    blob = storage.Blob.from_string(image_url, client=storage_client)
    image_bytes = blob.download_as_bytes()
    return Image.open(BytesIO(image_bytes))

def upload_image(image: Image, file_name: str):
    # Save image to a buffer
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    
    # Create a new blob in the bucket
    blob = bucket.blob(file_name)
    blob.upload_from_file(buffer, content_type="image/jpeg")
    return blob.public_url

def process_and_upload_image(file_data, image_id):
    try:
        # Convert the image data to a NumPy array
        nparr = np.frombuffer(file_data, np.uint8)
        img_lq = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_lq = img_lq.astype(np.float32) / 255.0
        processed_image = upscale(img_lq)

        # send image to cloud storage
        processed_image = (processed_image * 255.0).round().astype(np.uint8)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image_url = upload_image(processed_image, f"{image_id}.jpg")
        logging.info(f"Image processed and uploaded: {processed_image_url}")
    except Exception as e:
        logging.error(f"Error processing image in background: {e}")

@app.post("/process-image/")
async def process_image(image_url: ImageRequest):
    try:
        # Step 1: Download the image
        image = download_image(image_url.image_url)
        
        # Step 2: Manipulate the image (this is where you add your own logic)
        # API request the the model

        # Step 3: Upload the manipulated image
        transformed_image_url = upload_image(image, "transformed_image.jpg")
        return {"message": "Image processed successfully", "url": transformed_image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upscale/")
async def upscale_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format")

    image_id = str(uuid.uuid4())  # Generate a unique ID for the image process
    image_data = await file.read()
    # Schedule the processing of the image as a background task
    background_tasks.add_task(process_and_upload_image, image_data, image_id)

    # Immediately return a response to the client
    return {"message": "Image is being processed", "image_id": image_id}
