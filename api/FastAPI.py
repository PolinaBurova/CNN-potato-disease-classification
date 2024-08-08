

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

# Initialize the FastAPI app
app = FastAPI()


# Allow CORS (Cross-Origin Resource Sharing) for specific origins if we're making requests to the API from a frontend running on a different domain or port

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# Load the pre-trained Keras model
MODEL = tf.keras.models.load_model("C:/Users/polin/Downloads/potato-disease/models/1.keras")


# Define the class names corresponding to the model's output classes
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


# Simple GET endpoint to check if the API is running
@app.get("/ping")
async def ping():
    return "Hello, I am alive"

# Function to read the uploaded image file and convert it to a NumPy array
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# POST endpoint to receive an image file and return the model's prediction
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Read and preprocess the image
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0) # Expand dimensions to create a batch of size 1
    
     # Make a prediction using the model
    predictions = MODEL.predict(img_batch)

     # Determine the predicted class and its confidence level
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)