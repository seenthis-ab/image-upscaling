from fastapi import FastAPI, HTTPException, File, UploadFile
from google.cloud import storage
import os
from io import BytesIO
from PIL import Image
from pydantic import BaseModel


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
