from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from fastapi.responses import JSONResponse

app = FastAPI()
CLASS_NAMES = ['Potato_Early_blight',
               'Potato_Healthy',
               'Potato_Late_blight',
               'Sugarcane_Healthy',
               'Sugarcane_RedRot',
               'Sugarcane_RedRust']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Define a custom Keras model to load the SavedModel
class CustomModel(tf.keras.Model):
    def __init__(self, model_path):
        super(CustomModel, self).__init__()
        self.model = tf.saved_model.load(model_path)
        
    def call(self, inputs):
        return self.model(inputs)

# Load the model using CustomModel
MODEL = CustomModel("./models/1")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())

        # Resize the image
        width = 256
        height = 256
        resized_image = Image.fromarray(image).resize((width, height))
        resized_image = np.array(resized_image)

        # Convert image data type to float32
        resized_image = resized_image.astype(np.float32) / 255.0

        img_batch = np.expand_dims(resized_image, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        print(e)
        return JSONResponse(content={"status": "fail", "message": "Something went wrong"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=os.getenv("PORT", default=5000))
