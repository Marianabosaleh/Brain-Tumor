# Import necessary libraries
import os
import numpy as np
import collections
from PIL import Image
from sklearn.model_selection import train_test_split

# ================================
# ✅ Preprocess Images & Prepare Dataset
# ================================

# Define image size and dataset path
IMAGE_SIZE = (224, 224)  # Resize all images to 224x224
data_folder = "Data/brain_tumor"  # Path to dataset

X = []  # Image data
y = []  # Labels

# Assign numeric labels to categories
label_mapping = {"1": 0, "2": 1, "3": 2}

# Load and preprocess images
for label in os.listdir(data_folder):
    label_path = os.path.join(data_folder, label)

    if os.path.isdir(label_path):  # Ensure it's a directory
        for file in os.listdir(label_path):
            if file.endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(label_path, file)

                # Open image, convert to RGB, resize, and normalize
                image = Image.open(file_path).convert("RGB").resize(IMAGE_SIZE)
                image_array = np.array(image) / 255.0  # Normalize pixel values

                X.append(image_array)
                y.append(label_mapping[label])  # Convert label to integer

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Flatten images for classical ML models
X_train_flatten = X_train.reshape(X_train.shape[0], -1)
X_test_flatten = X_test.reshape(X_test.shape[0], -1)

print(f"New shape of training data: {X_train_flatten.shape}")  # Should be (samples, 150528)
print(f"New shape of testing data: {X_test_flatten.shape}")

# Save preprocessed data (optional for future runs)
np.save("X_train.npy", X_train_flatten)
np.save("X_test.npy", X_test_flatten)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Data Preprocessing Complete!")
