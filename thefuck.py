import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import time
import cv2
from picamera2 import Picamera2, Preview
from libcamera import Transform
from pynput.keyboard import Key, Listener

location = '/home/phenom/Pictures/test.jpg'
picam2 = Picamera2()
camera_config = picam2.create_preview_configuration()
picam2.configure(camera_config)
picam2.start_preview(Preview.QTGL, transform=Transform(hflip=1))
picam2.start()

# Load the VGG16 model with ImageNet weights
from tensorflow.keras.applications.vgg16 import VGG16
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Function to preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize image to match VGG16 input shape
    img = image.img_to_array(img)  # Convert image to array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess input (normalize pixels)
    return img

# Function to perform object classification
def classify_image(img):
    preprocessed_img = preprocess_image(img)
    features = base_model.predict(preprocessed_img)  # Extract features using VGG16
    # Here you can add your own classification logic using the features extracted
    # For example, you can add fully connected layers on top of VGG16 and train it on your own dataset
    return features

# Function to capture image
def capture_image():
    print('wtf')
    picam2.capture_file(location)
    img = cv2.imread(location)
    features = classify_image(img)
    print('wtf')
    print(help(cv2))
    # Process features as needed
    
    # Add labels and classifications to the image
    #cv2.putText(img, "Predicted Label", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #cv2.putText(img, "Confidence", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # Replace "Predicted Label" and "Confidence" with actual label and confidence values
    
    # Save the classified image as out.jpg
    cv2.imwrite('/home/phenom/Pictures/out.jpg', img)

# Callback function for key press
def on_press(key):
    if key == Key.space:
        capture_image()

# Start listening for key press events
with Listener(on_press=on_press) as listener:
    listener.join()

# Continuous loop for keeping the script running
while True:
    time.sleep(1)
