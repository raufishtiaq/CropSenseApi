from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

MODEL = tf.keras.models.load_model('./models/1')
CLASS_NAMES = ['Potato_Early_blight',
 'Potato_Healthy',
 'Potato_Late_blight',
 'Sugarcane_Healthy',
 'Sugarcane_RedRot',
 'Sugarcane_RedRust']

@app.get("/ping")
async def ping():
    return "Hello, I'm alive."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=os.getenv("PORT", default=5000))
