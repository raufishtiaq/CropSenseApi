from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.responses import JSONResponse

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the class names for the predictions
CLASS_NAMES = ['Potato_Early_blight',
               'Potato_Healthy',
               'Potato_Late_blight',
               'Sugarcane_Healthy',
               'Sugarcane_RedRot',
               'Sugarcane_RedRust']

# Load the TensorFlow model
model_path = "./models/1"
model = tf.saved_model.load(model_path)

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    # Read the uploaded file as an image
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image = read_file_as_image(await file.read())

        # Resize the image to match the model input size
        width = 256
        height = 256
        resized_image = Image.fromarray(image).resize((width, height))
        resized_image = np.array(resized_image)

        # Prepare the image for prediction
        img_batch = np.expand_dims(resized_image, 0)

        # Perform inference using the loaded model
        predictions = model(img_batch)

        # Get the predicted class and confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            'class': predicted_class,
            'confidence': confidence
        }
    except Exception as e:
        print(e)
        return JSONResponse(content={"status": "fail", "message": "Something went wrong"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
