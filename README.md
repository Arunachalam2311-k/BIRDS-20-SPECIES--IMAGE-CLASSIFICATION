## BIRDS 20 SPECIES - IMAGE CLASSIFICATION

# Project Overview

This project involves the classification of 20 different bird species using a Convolutional Neural Network (CNN) model. The application is built with Streamlit, allowing users to upload an image of a bird and get the predicted species name. The model is trained on a high-quality dataset containing 3208 training images, 100 test images, and 100 validation images.

# Dataset Description

Total Species: 20

Image Format: JPG (224x224x3)

Training Images: 3208

Testing Images: 100 (5 per species)

Validation Images: 100 (5 per species)

Structure: The dataset includes subdirectories for each species and is organized into train, test, and valid folders.

CSV File: birds.csv containing:

filepaths: Relative file paths

labels: Species names

scientific name: Latin scientific names

dataset: Indicates the set (train/test/valid)

class_id: Index of the class

# Model

Model Type: CNN

Framework: TensorFlow / Keras

Input Size: 150x150x3 (for faster training)

Output: Predicted class label with confidence score

Streamlit App Features

Upload images in .jpg, .jpeg, or .png formats

Display uploaded image and prediction result

Predicts species from 20 predefined classes

How to Run

# Install required packages:

pip install streamlit tensorflow pillow numpy matplotlib

Run the Streamlit app:

streamlit run app.py

Model Loading

The model is loaded from a saved .h5 file located at D:/work/my_cnn_model.h5. Ensure the file exists and is accessible from the provided path.

# Notes

Images are preprocessed by resizing to 150x150 and normalizing pixel values to [0, 1].

The application uses session state to maintain the uploaded image and prediction.

The dataset is skewed with ~80% images of male birds which may affect generalization on female species.

# Acknowledgments

The dataset is curated from internet image searches and refined for quality and uniqueness.

All images are original with no augmentations.

Developed as part of an image classification project for identifying bird species using deep learning and Streamlit.
