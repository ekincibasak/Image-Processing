import numpy as np  # Import the numpy library for linear algebra operations
import pandas as pd  # Import the pandas library for data processing and CSV file I/O

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import cv2  # Import the OpenCV library for computer vision tasks
from glob import glob  # Import the glob function to search for files  # Enable inline plotting for Matplotlib
import matplotlib.pyplot as plt  # Import Matp

X_shape = 224  # Define the image shape

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def process():
    # Define the path to your image directory
    image_path = "/home/Share/Safak/MontgomerySet10GB/Lung Segmentation/CXR_png"

    # Create empty lists to store images and labels for both datasets
    images_montgomery = []  # To store images from the Montgomery dataset
    labels_montgomery = []  # To store labels for the Montgomery dataset
    images_china = []       # To store images from the ChinaCXR dataset
    labels_china = []       # To store labels for the ChinaCXR dataset

    # Function to load and preprocess images from a directory
    def load_images_from_directory(directory, labels, dataset_label):
        image_files = os.listdir(directory)
        for file_name in image_files:
            if file_name.startswith('MCUCXR'):
                file_dataset_label = 'Montgomery'
            elif file_name.startswith('CHNCXR'):
                file_dataset_label = 'ChinaCXR'
            else:
                continue  # Skip files that don't belong to Montgomery or ChinaCXR
            
            if file_dataset_label == dataset_label:
                if file_name.endswith('0.png'):
                    labels.append(0)  # Normal
                elif file_name.endswith('1.png'):
                    labels.append(1)  # Abnormal
                else:
                    continue  # Skip files that don't have labels

                # Load and preprocess the image
                image = cv2.resize(cv2.imread(os.path.join(directory, file_name)), (X_shape, X_shape))[:, :, 0]
                
                # Append the image to the appropriate dataset
                if dataset_label == 'Montgomery':
                    images_montgomery.append(image)
                elif dataset_label == 'ChinaCXR':
                    images_china.append(image)

        # Load and preprocess images from the directory for both datasets
    load_images_from_directory(image_path, labels_montgomery, 'Montgomery')
    load_images_from_directory(image_path, labels_china, 'ChinaCXR')

        # Convert the lists to NumPy arrays
    images_montgomery = np.array(images_montgomery)
    labels_montgomery = np.array(labels_montgomery)
    images_china = np.array(images_china)
    labels_china = np.array(labels_china)

        # Count the number of normal (0) and abnormal (1) samples for both datasets
    num_normal_montgomery = np.count_nonzero(labels_montgomery == 0)
    num_abnormal_montgomery = np.count_nonzero(labels_montgomery == 1)
    num_normal_china = np.count_nonzero(labels_china == 0)
    num_abnormal_china = np.count_nonzero(labels_china == 1)

        # Print the statistics for both datasets
    print("Montgomery Dataset:")
    print("Number of Normal (0) Samples:", num_normal_montgomery)
    print("Number of Abnormal (1) Samples:", num_abnormal_montgomery)

    print("\nChinaCXR Dataset:")
    print("Number of Normal (0) Samples:", num_normal_china)
    print("Number of Abnormal (1) Samples:", num_abnormal_china)

    
    return images_montgomery,labels_montgomery,images_china,labels_china,num_normal_montgomery,num_abnormal_montgomery,num_normal_china,num_abnormal_china

