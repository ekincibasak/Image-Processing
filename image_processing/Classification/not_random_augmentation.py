import cv2
import numpy as np
import os

# Define a function to perform data augmentation
def augment_data(images, labels, num_augmented_samples, save_dir=None, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)  # Set the random seed for NumPy

    augmented_images = []
    augmented_labels = []

    # Create a fixed list of indices for image selection
    num_original_samples = len(images)
    index_list = list(range(num_original_samples))
    np.random.seed(random_seed)  # Set the seed for shuffling
    np.random.shuffle(index_list)

    for i in range(num_augmented_samples):
        # Randomly select an image index from the shuffled list
        idx = index_list[i % num_original_samples]  # Wrap around to reuse indices
        image = images[idx]
        label = labels[idx]

        # Apply random augmentation techniques
        if np.random.choice([True, False]):
            # Flip horizontally with 50% probability
            image = np.fliplr(image)

        if np.random.choice([True, False]):
            # Rotate the image by a random angle between -15 and 15 degrees
            angle = np.random.uniform(-15, 15)
            rows, cols = image.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            image = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if np.random.choice([True, False]):
            # Adjust brightness by scaling pixel values
            alpha = np.random.uniform(0.7, 1.3)  # Brightness factor
            image = cv2.multiply(image, np.array([alpha]))

        augmented_images.append(image)
        augmented_labels.append(label)

        # Save the augmented image if save_dir is provided
        if save_dir:
            file_name = f"augmented_{i}_{label}.png"
            cv2.imwrite(os.path.join(save_dir, file_name), image)

    return np.array(augmented_images), np.array(augmented_labels)

# Define the number of augmented samples you want to generate
num_augmented_samples = 600  # Adjust this number as needed

# Directory to save augmented data
save_dir = '/kaggle/working/augmented_data'  # Replace with your desired directory

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Set a fixed random seed for reproducibility
random_seed = 42

# Perform data augmentation on the Montgomery dataset
augmented_images_montgomery, augmented_labels_montgomery = augment_data(
    images_montgomery, labels_montgomery, num_augmented_samples, save_dir, random_seed)

# Calculate the counts of new healthy and unhealthy Montgomery data samples
num_new_normal_montgomery = np.count_nonzero(augmented_labels_montgomery == 0)
num_new_abnormal_montgomery = np.count_nonzero(augmented_labels_montgomery == 1)

print("Montgomery Dataset:")
print(f"Number of Normal (0) Samples Before Augmentation: {num_normal_montgomery}")
print(f"Number of Abnormal (1) Samples Before Augmentation: {num_abnormal_montgomery}")
print(f"Number of New Normal (0) Samples After Augmentation: {num_new_normal_montgomery}")
print(f"Number of New Abnormal (1) Samples After Augmentation: {num_new_abnormal_montgomery}")