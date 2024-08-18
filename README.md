# CNN-potato-disease-classification

This project focuses on the classification of potato leaf diseases using Convolutional Neural Networks (CNNs). The model classifies potato leaf images into one of three categories: Early Blight, Late Blight, or Healthy. The dataset used for this project is the PlantVillage dataset, which provides a rich collection of labeled images for training and evaluation.  

## Model Training

The model was built using TensorFlow and trained on the PlantVillage dataset. 
https://www.kaggle.com/datasets/arjuntejaswi/plant-village

### Key Steps:
1. **Data Loading:** The dataset is loaded from the `PlantVillage` directory.
2. **Preprocessing:** The images are resized, normalized, and augmented as needed. This preprocessing is done in the `notebook` directory.
3. **Model Building:** A Convolutional Neural Network (CNN) is constructed using TensorFlow’s Keras API, within the `api` directory.
4. **Training:** The model is trained on the dataset, and the trained model is saved to the `models` directory.

The architecture includes sequential layers for resizing, rescaling, and augmenting the input images, followed by several convolutional layers with ReLU activation and max-pooling. The final output layer uses softmax activation to predict the class probabilities.

### Model Evaluation

After training, the model was evaluated on a test set, and the results showed a high accuracy (around 99%) in classifying the potato leaf diseases. Below are some of the predictions made by the model:

![Potato Disease Predictions](https://github.com/PolinaBurova/CNN-potato-disease-classification/blob/main/notebook/potato-disease-prediction.png)  


## Model Deployment

The trained model is deployed using FastAPI. The API can accept an image and return the predicted class label.

### FastAPI Implementation (`api/FastAPI.py`):
- Loads the trained model from the `models` directory.
- Provides an endpoint `/predict` to handle image uploads and predictions.
- Returns the predicted class label (`Early Blight`, `Late Blight`, or `Healthy`).

![API Predict](https://github.com/PolinaBurova/CNN-potato-disease-classification/blob/main/api/FastAPIScreenshot1.png)  

