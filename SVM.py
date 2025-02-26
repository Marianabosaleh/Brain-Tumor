# ================================
# ✅ Step 1: Load Preprocessed Data
# ================================

import numpy as np

# Load preprocessed dataset
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# ================================
# ✅ Fix Unicode Font Issue in Matplotlib
# ================================
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Set a font that supports Unicode
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["axes.unicode_minus"] = False

# Optionally suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ================================
# ✅ Step 1.5: Apply PCA for Feature Reduction
# ================================
from sklearn.decomposition import PCA

print("\n✅ Applying PCA for Feature Reduction...")
pca = PCA(n_components=500)  # Reduce to 500 features
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"New Shape After PCA: {X_train_pca.shape}, {X_test_pca.shape}")

# ================================
# ✅ Step 2: Train Support Vector Machine (SVM)
# ================================

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("\n✅ Training SVM Without Tuning...")
svm_model = SVC(kernel="linear")  # Using Linear Kernel
svm_model.fit(X_train_pca, y_train)

# Predictions
y_pred_svm = svm_model.predict(X_test_pca)

# Evaluate Model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"\n📌 SVM Accuracy Without Tuning: {accuracy_svm:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix
print("\nConfusion Matrix:")
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_svm)

# ================================
# ✅ Step 3: Hyperparameter Tuning using GridSearchCV
# ================================

from sklearn.model_selection import GridSearchCV

print("\n✅ Performing Hyperparameter Tuning (This May Take Some Time)...")

# Define reduced hyperparameter grid for faster execution
param_grid = {
    "C": [1, 10],  # Reduced to speed up tuning
    "kernel": ["linear", "rbf"],  # Avoid "poly" as it's slow
    "gamma": ["scale"]  # Using default gamma
}

# Perform Grid Search with Cross-Validation
svm_grid = GridSearchCV(SVC(), param_grid, cv=3, n_jobs=-1, verbose=1)
svm_grid.fit(X_train_pca, y_train)

# Get best model and parameters
best_svm = svm_grid.best_estimator_
print(f"\n✅ Best Parameters: {svm_grid.best_params_}")

# Make predictions with best model
y_pred_best_svm = best_svm.predict(X_test_pca)

# Evaluate Optimized Model
accuracy_best_svm = accuracy_score(y_test, y_pred_best_svm)
print(f"\n📌 Best SVM Accuracy: {accuracy_best_svm:.4f}")

print("\nOptimized Classification Report:")
print(classification_report(y_test, y_pred_best_svm))

print("\nOptimized Confusion Matrix:")
cm_best_svm = confusion_matrix(y_test, y_pred_best_svm)
print(cm_best_svm)

# ================================
# ✅ Step 4: Compare Performance Before & After Tuning
# ================================

print(f"\nSVM Accuracy Before Tuning: {accuracy_svm:.4f}")
print(f"SVM Accuracy After Tuning: {accuracy_best_svm:.4f}")

# ================================
# ✅ Step 5: Visualization (Confusion Matrix)
# ================================

# Plot Confusion Matrix for Optimized SVM Model
plt.figure(figsize=(6,5))
sns.heatmap(cm_best_svm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - SVM")
plt.show()

# ================================
# ✅ Step 6: Precision-Recall Curve
# ================================

from sklearn.metrics import precision_recall_curve

# Get precision and recall values for each class
precision = dict()
recall = dict()

for i in range(3):  # Assuming 3 classes (0, 1, 2)
    precision[i], recall[i], _ = precision_recall_curve((y_test == i).astype(int), (y_pred_best_svm == i).astype(int))

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - SVM")
plt.legend(loc="best")
plt.show()

# ================================
# ✅ Step 7: Display Sample Misclassified Images
# ================================

import random

# Get indices of misclassified images
misclassified_idxs = np.where(y_pred_best_svm != y_test)[0]

# Randomly select a few misclassified images
random_samples = random.sample(list(misclassified_idxs), min(6, len(misclassified_idxs)))  # Ensure valid sample size

plt.figure(figsize=(12, 8))

for i, idx in enumerate(random_samples):
    plt.subplot(2, 3, i + 1)
    img = X_test[idx].reshape(224, 224, 3)  # Reshape to original size
    plt.imshow(img)
    plt.title(f"True: {y_test[idx]}, Predicted: {y_pred_best_svm[idx]}")
    plt.axis("off")

plt.suptitle("Sample Misclassified Images - SVM", fontsize=14)
plt.show()

# End of the script
print("\n✅ SVM Training & Visualization Completed!")
