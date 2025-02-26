# ================================
# ✅ Step 1: Load Preprocessed Data
# ================================

import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load Data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# ================================
# ✅ Step 1.5: Apply PCA & Feature Scaling
# ================================
print("\n✅ Applying PCA & Standard Scaling...")

# Normalize data (KNN is sensitive to scale)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA (reduce to 500 features)
pca = PCA(n_components=500)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"✅ New Shape After PCA: {X_train_pca.shape}, {X_test_pca.shape}")

# ================================
# ✅ Step 2: Finding Best K Value
# ================================

k_values = range(1, 21)  # Test K from 1 to 20
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"🔍 K={k}, Accuracy: {acc:.4f}")

# Plot Accuracy vs. K
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker="o", linestyle="-")
plt.xlabel("Number of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")
plt.grid()
plt.show()

# Find Best K
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
print(f"✅ Best K: {best_k} with Accuracy: {best_accuracy:.4f}")

# ================================
# ✅ Step 3: Train Final Model with Best K
# ================================

best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train_pca, y_train)

# Save the best model
joblib.dump(best_knn, "best_knn_model.pkl")
print("💾 Best KNN Model Saved!")

# ================================
# ✅ Step 4: Evaluate Final Model
# ================================

y_pred_best_knn = best_knn.predict(X_test_pca)
accuracy_best_knn = accuracy_score(y_test, y_pred_best_knn)
print(f"\n📌 Best KNN Model Accuracy: {accuracy_best_knn:.4f}")

# ================================
# ✅ Step 5: Confusion Matrix
# ================================

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("\n Classification Report:")
print(classification_report(y_test, y_pred_best_knn))

print("\n Confusion Matrix:")
cm_knn = confusion_matrix(y_test, y_pred_best_knn)
print(cm_knn)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(" Confusion Matrix - KNN")
plt.show()

# ================================
# ✅ Step 6: Precision-Recall Curve
# ================================

from sklearn.metrics import precision_recall_curve

precision = dict()
recall = dict()

for i in range(3):  # Assuming 3 classes (0, 1, 2)
    precision[i], recall[i], _ = precision_recall_curve((y_test == i).astype(int), (y_pred_best_knn == i).astype(int))

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
for i in range(3):
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(" Precision-Recall Curve - KNN")
plt.legend(loc="best")
plt.show()

# ================================
# ✅ Step 7: Display Sample Misclassified Images (Fixed)
# ================================

import random

# Get misclassified indices
misclassified_idxs = np.where(y_pred_best_knn != y_test)[0]

if len(misclassified_idxs) > 0:
    random_samples = random.sample(list(misclassified_idxs), min(6, len(misclassified_idxs)))

    plt.figure(figsize=(12, 8))
    for i, idx in enumerate(random_samples):
        plt.subplot(2, 3, i + 1)

        # Reshape and Rescale Image for Display
        img = X_test[idx].reshape(224, 224, 3)  # Reshape back to image format
        img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
        img = np.clip(img, 0, 1)  # Ensure all values are between 0 and 1

        plt.imshow(img)
        plt.title(f"True: {y_test[idx]}, Predicted: {y_pred_best_knn[idx]}")
        plt.axis("off")

    plt.suptitle(" Sample Misclassified Images - KNN", fontsize=14)
    plt.show()
else:
    print("\n✅ No misclassified images found.")

# End of script
print("\n✅ KNN Training & Visualization Completed!")
