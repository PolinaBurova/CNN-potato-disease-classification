# CNN-potato-disease-classification

## Model Training

The model was built using TensorFlow and trained on the PlantVillage dataset. 
https://www.kaggle.com/datasets/arjuntejaswi/plant-village

It is a Convolutional Neural Network (CNN) that classifies potato leaf images into one of the three categories: Potato_Early_blight; Potato_Late_blight and Potato_healthy.

### Key Steps:
1. **Data Loading**: The dataset is loaded from the `PlantVillage` directory.
2. **Preprocessing**: The images are resized, normalized, and augmented as needed.
3. **Model Building**: A CNN is constructed using TensorFlow's Keras API.
4. **Training**: The model is trained on the dataset, and the trained model is saved to the `models` directory.

## Model Deployment

The trained model is deployed using FastAPI. The API can accept an image and return the predicted class label.

### FastAPI Implementation (`api/FastAPI.py`):
- Loads the trained model from the `models` directory.
- Provides an endpoint `/predict` to handle image uploads and predictions.
- Returns the predicted class label (`Early Blight`, `Late Blight`, or `Healthy`).

