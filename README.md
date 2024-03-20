# Image Processing for Lung X-Ray Classification

## Overview
This repository contains code for processing Montgomery X-ray lung images for the purpose of classifying them as healthy or unhealthy. Various deep learning models such as AlexNet, ResNet, and DenseNet are implemented for classification tasks. Additionally, preprocessing techniques including augmentation and histogram equalization are applied to enhance the performance of the models.

## Dataset
The dataset used in this project consists of Montgomery X-ray lung images. These images are categorized into two classes: healthy and unhealthy.

## Models
The following deep learning models are implemented in this repository:
- AlexNet
- ResNet
- DenseNet

## Preprocessing Techniques
The following preprocessing techniques are applied to the images:
- Augmentation: Various augmentation techniques such as rotation, flipping, and scaling are employed to increase the diversity of the dataset and improve model generalization.
- Histogram Equalization: This technique is used to enhance the contrast of the images, making it easier for the models to extract meaningful features.

## File Structure
- `preprocessing/`: Contains scripts for preprocessing the images.
- `models/`: Contains implementations of the deep learning models.
- `utils/`: Contains utility functions used throughout the project.
- `train.py`: Script for training the deep learning models.
- `evaluate.py`: Script for evaluating the trained models.

## Usage
1. Clone this repository:
