import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import itertools
import pickle  # Import the pickle module


"""
# Combine original and augmented data for both images and labels
X_combined = np.concatenate([images_montgomery, images_china, augmented_images_montgomery], axis=0)
y_combined = np.concatenate([labels_montgomery, labels_china, augmented_labels_montgomery], axis=0)
# Check the shape of X_combined and y_combined
print("Shape of X_combined:", X_combined.shape)
print("Shape of y_combined:", y_combined.shape)

# Check the size (total number of elements) of X_combined and y_combined
print("Size of X_combined:", X_combined.size)
print("Size of y_combined:", y_combined.size)


# Normalize the X (image) data
X_combined_normalized = X_combined / 255.0  # Scale to the [0, 1] range

# Normalize the Y (label) data (assuming it's binary)
Y_combined_normalized = y_combined  # No need to normalize labels for binary classification

# Check the shape of the normalized data
print("Shape of X_combined_normalized:", X_combined_normalized.shape)
print("Shape of Y_combined_normalized:", Y_combined_normalized.shape)

# Check the minimum and maximum pixel values of the normalized X data
min_pixel_value = np.min(X_combined_normalized)
max_pixel_value = np.max(X_combined_normalized)
print("Minimum Pixel Value:", min_pixel_value)
print("Maximum Pixel Value:", max_pixel_value)

# Assuming you have your data (X_combined_normalized, Y_combined_normalized)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_combined_normalized, Y_combined_normalized, test_size=0.2, random_state=42)
# Print the shapes of the sets
print(f"Training Data - X_train: {X_train.shape}")
print(f"Training Data - y_train: {y_train.shape}")
print(f"Test Data - X_test: {X_test.shape}")
print(f"Test Data - y_test: {y_test.shape}")"""


def train_test(images_montgomery, images_china, augmented_images_montgomery,labels_montgomery, labels_china, augmented_labels_montgomery):
    X_combined = np.concatenate([images_montgomery, images_china, augmented_images_montgomery], axis=0)
    y_combined = np.concatenate([labels_montgomery, labels_china, augmented_labels_montgomery], axis=0)
    # Check the shape of X_combined and y_combined
    print("Shape of X_combined:", X_combined.shape)
    print("Shape of y_combined:", y_combined.shape)

    # Check the size (total number of elements) of X_combined and y_combined
    print("Size of X_combined:", X_combined.size)
    print("Size of y_combined:", y_combined.size)


    # Normalize the X (image) data
    X_combined_normalized = X_combined / 255.0  # Scale to the [0, 1] range

    # Normalize the Y (label) data (assuming it's binary)
    Y_combined_normalized = y_combined  # No need to normalize labels for binary classification

    # Check the shape of the normalized data
    print("Shape of X_combined_normalized:", X_combined_normalized.shape)
    print("Shape of Y_combined_normalized:", Y_combined_normalized.shape)

    # Check the minimum and maximum pixel values of the normalized X data
    min_pixel_value = np.min(X_combined_normalized)
    max_pixel_value = np.max(X_combined_normalized)
    print("Minimum Pixel Value:", min_pixel_value)
    print("Maximum Pixel Value:", max_pixel_value)

    # Assuming you have your data (X_combined_normalized, Y_combined_normalized)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_combined_normalized, Y_combined_normalized, test_size=0.2, random_state=42)
    # Print the shapes of the sets
    print(f"Training Data - X_train: {X_train.shape}")
    print(f"Training Data - y_train: {y_train.shape}")
    print(f"Test Data - X_test: {X_test.shape}")
    print(f"Test Data - y_test: {y_test.shape}")

    return X_train, X_test, y_train, y_test