# ================================
# ✅ Step 1: Load Preprocessed Data
# ================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed dataset
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# ================================
# ✅ Step 1.5: Apply PCA for Feature Reduction
# ================================
from sklearn.decomposition import PCA

print("\n✅ Applying PCA for Feature Reduction...")
pca = PCA(n_components=500)  # Reduce to 500 features
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"✅ New Shape After PCA: {X_train_pca.shape}, {X_test_pca.shape}")

# ================================
# ✅ Step 2: Train Logistic Regression Model
# ================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, solver='lbfgs')

# Train model
log_reg.fit(X_train_pca, y_train)

# Make predictions
y_pred_log = log_reg.predict(X_test_pca)

# Evaluate model
accuracy_log = accuracy_score(y_test, y_pred_log)
print(f"\n📌 Logistic Regression Accuracy: {accuracy_log:.4f}")

# Classification Report
print("\n Classification Report:")
print(classification_report(y_test, y_pred_log))

# Confusion Matrix
print("\n Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_log)
print(cm)

# ================================
# ✅ Step 3: Hyperparameter Tuning using GridSearchCV
# ================================

from sklearn.model_selection import GridSearchCV

print("\n✅ Performing Hyperparameter Tuning (This may take time)...")

# Define hyperparameter grid
param_grid = {
    "C": [0.1, 1, 10],  # Reduced for faster tuning
    "solver": ["liblinear", "lbfgs"]  # Removed "saga" (too slow)
}

# Perform Grid Search with Cross-Validation
log_reg_grid = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=3, n_jobs=-1, verbose=1)
log_reg_grid.fit(X_train_pca, y_train)

# Get best model and parameters
best_log_reg = log_reg_grid.best_estimator_
print(f"\n✅ Best Parameters: {log_reg_grid.best_params_}")

# Make predictions with best model
y_pred_best_log = best_log_reg.predict(X_test_pca)

# Evaluate Optimized Model
accuracy_best_log = accuracy_score(y_test, y_pred_best_log)
print(f"\n📌 Best Logistic Regression Accuracy: {accuracy_best_log:.4f}")

print("\n Optimized Classification Report:")
print(classification_report(y_test, y_pred_best_log))

print("\n Optimized Confusion Matrix:")
cm_best = confusion_matrix(y_test, y_pred_best_log)
print(cm_best)

# ================================
# ✅ Step 4: Compare Performance Before & After Tuning
# ================================

print(f"\n Logistic Regression Accuracy Before Tuning: {accuracy_log:.4f}")
print(f" Logistic Regression Accuracy After Tuning: {accuracy_best_log:.4f}")

# ================================
# ✅ Step 5: Visualization (Confusion Matrix)
# ================================

# Plot Confusion Matrix for Optimized Model
plt.figure(figsize=(6,5))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(" Confusion Matrix - Logistic Regression")
plt.show()
plt.close()

# ================================
# ✅ Step 6: Precision-Recall Curve
# ================================

from sklearn.metrics import precision_recall_curve

# Get precision and recall values for each class
precision = dict()
recall = dict()

for i in range(3):  # Assuming 3 classes (0, 1, 2)
    precision[i], recall[i], _ = precision_recall_curve((y_test == i).astype(int), (y_pred_best_log == i).astype(int))

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Logistic Regression")
plt.legend(loc="best")
plt.show()
plt.close()

# ================================
# ✅ Step 7: Display Sample Misclassified Images
# ================================

import random

# Get indices of misclassified images
misclassified_idxs = np.where(y_pred_best_log != y_test)[0]

if len(misclassified_idxs) > 0:
    # Randomly select a few misclassified images
    random_samples = random.sample(list(misclassified_idxs), min(6, len(misclassified_idxs)))  # Ensure valid sample size

    plt.figure(figsize=(12, 8))

    for i, idx in enumerate(random_samples):
        plt.subplot(2, 3, i + 1)
        img = X_test[idx].reshape(224, 224, 3)  # Reshape to original size
        plt.imshow(img)
        plt.title(f"True: {y_test[idx]}, Predicted: {y_pred_best_log[idx]}")
        plt.axis("off")

    plt.suptitle(" Sample Misclassified Images - Logistic Regression", fontsize=14)
    plt.show()
    plt.close()
else:
    print("\n✅ No misclassified images found.")

# End of the script
print("\n✅ Logistic Regression Training & Visualization Completed!")
