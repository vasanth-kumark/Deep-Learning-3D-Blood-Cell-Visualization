from keras.models import load_model # type: ignore
from keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
import cv2
import tensorflow as tf

# Load the pre-trained model
model = load_model('model.h5')  # Use the correct model name

# Function to load and preprocess the image for prediction
def prepare_image(image_path):
    image = load_img(image_path, target_size=(224, 224))  # Adjust size to 224x224
    image = img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Use the same preprocessing function
    image = np.expand_dims(image, axis=0)  # Expand dimensions for batch
    return image

# Predict the blood cell type
def predict_blood_cell(image_path, classes):
    image = prepare_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=-1)
    return classes[predicted_class[0]]  # Return the predicted class name

if __name__ == "__main__":
    # Define the class labels based on your model
    classes = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    
    # Test the prediction
    predicted_class = predict_blood_cell('S:\STUDIES\PROJECTS\Deep Learning Based 3D Reconstruction of Blood Cells\misc\BloodImage_00104.jpg', classes)
    print(f'Predicted class: {predicted_class}')
