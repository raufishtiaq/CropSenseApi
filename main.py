
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
    uvicorn.run(app, host='localhost', port=8000)

# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# from fastapi.responses import JSONResponse

# app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MODEL = tf.keras.models.load_model("./models/1")
# # MODEL = tf.keras.layers.TFSMLayer("./models/1")
# CLASS_NAMES = ['Potato_Early_blight',
#                'Potato_Healthy',
#                'Potato_Late_blight',
#                'Sugarcane_Healthy',
#                'Sugarcane_RedRot',
#                'Sugarcane_RedRust']


# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"


# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         image = read_file_as_image(await file.read())

#         # Resize the image
#         width = 256
#         height = 256
#         resized_image = Image.fromarray(image).resize((width, height))
#         resized_image = np.array(resized_image)

#         img_batch = np.expand_dims(resized_image, 0)

#         predictions = MODEL.predict(img_batch)

#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = np.max(predictions[0])
#         return {
#             'class': predicted_class,
#             'confidence': float(confidence)
#         }
#     except Exception as e:
#         print(e)
#         return JSONResponse(content={"status": "fail", "message": "Something went wrong"}, status_code=500)

# if __name__ == "__main__":
#     uvicorn.run(app, host='localhost', port=8000)
