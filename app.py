import easyocr
import base64
import numpy as np
import cv2
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model once at startup
reader = easyocr.Reader(['en'])

class ImageData(BaseModel):
    image_base64: str

@app.post("/ocr")
def ocr(data: ImageData):
    try:
        # Decode base64 image
        img_bytes = base64.b64decode(data.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run OCR
        result = reader.readtext(img, detail=0)

        return {
            "success": True,
            "text": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

