from fastapi import FastAPI, HTTPException, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from google.cloud import storage
from image_upscaling import upscale
import os
import numpy as np
import cv2
import uuid
from io import BytesIO
from PIL import Image
from pydantic import BaseModel
from typing import Dict

import logging
import json

logging.basicConfig(level=logging.INFO)


app = FastAPI()

# Specify your CORS configuration here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"]   # Allows all headers
)

app.mount("/static", StaticFiles(directory="static"), name="static")

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

def upload_image(image: Image, file_name: str, ext: str = "jpeg"):

    # Convert ext to uppercase
    ext_upper = ext.upper()
    # Save image to a buffer
    buffer = BytesIO()
    image.save(buffer, format=ext_upper, quality=95)
    buffer.seek(0)
    
    # Create a new blob in the bucket
    blob = bucket.blob(file_name)
    blob.upload_from_file(buffer, content_type=f"image/{ext}")
    return blob.public_url

def process_and_upload_image(file_data, img_name, img_ext, image_id):
    try:
        nparr = np.frombuffer(file_data, np.uint8)
        img_lq = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_lq = img_lq.astype(np.float32) / 255.0
        processed_image = upscale(img_lq)

        logging.info(f"Start uploading image: {img_name}.{img_ext}")
        processed_image_url = upload_image(processed_image, f"{img_name}.{img_ext}", img_ext)
        logging.info(f"Image processed and uploaded: {processed_image_url}")
        
        # Ensure the message sent to the client includes the URL in the expected format
        message = {"url": processed_image_url}
        #await manager.send_message(image_id, json.dumps(message))
        return message
    except Exception as e:
        logging.error(f"Error processing image in background: {e}")
        error_message = {"error": "Error processing image"}
        #await manager.send_message(image_id, json.dumps(error_message))

@app.get("/")
def read_root():
    return FileResponse(os.path.join('static', 'index.html'))

    
@app.post("/upscale/")
async def upscale_image(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    # Get format of the image
    img_ext = file.filename.split(".")[-1]

    # Get image name
    img_name = file.filename.split(".")[0]

    logging.info(f"Processing image: {img_name}.{img_ext}")

    image_id = str(uuid.uuid4())  # Generate a unique ID for the image process
    image_data = await file.read()

    message = process_and_upload_image(image_data, img_name, img_ext, image_id)
    # Immediately return a response to the client
    return {"message": "Image is being processed", "id": image_id, "url": message["url"]}
