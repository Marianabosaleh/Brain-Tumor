# ================================
# ✅ Step 1: Load Preprocessed Data
# ================================

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load Data
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

print(f"✅ Loaded Data: X_train: {X_train.shape}, X_test: {X_test.shape}")

# ================================
# ✅ Step 1.5: Apply PCA & Feature Scaling
# ================================
print("\n✅ Applying PCA & Standard Scaling...")

# Normalize data (Decision Trees & RF are less sensitive to scaling, but still helps)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA (reduce to 500 features)
pca = PCA(n_components=500)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"✅ New Shape After PCA: {X_train_pca.shape}, {X_test_pca.shape}")

# ================================
# ✅ Step 2: Train & Tune Decision Tree Model
# ================================

print("🚀 Training & Optimizing Decision Tree...")

# Define hyperparameter grid
param_grid_dt = {
    "max_depth": [10, 20, None],  # Try different depths
    "min_samples_split": [2, 5, 10],  # Try different splits
}

# Perform Grid Search for Decision Tree
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=3, n_jobs=-1, verbose=1)
dt_grid.fit(X_train_pca, y_train)

# Get best model and parameters
best_dt = dt_grid.best_estimator_
print(f"✅ Best Decision Tree Parameters: {dt_grid.best_params_}")

# Predict using best Decision Tree
y_pred_dt = best_dt.predict(X_test_pca)

# Evaluate Model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"🌲 Best Decision Tree Accuracy: {accuracy_dt:.4f}")
print("\n📊 Classification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt))

# Save the best Decision Tree Model
joblib.dump(best_dt, "best_decision_tree.pkl")
print("💾 Best Decision Tree Model Saved!")

# ================================
# ✅ Step 3: Train & Tune Random Forest Model
# ================================

print("🚀 Training & Optimizing Random Forest...")

# Define hyperparameter grid
param_grid_rf = {
    "n_estimators": [50, 100, 200],  # Number of trees
    "max_depth": [10, 20, None],  # Tree depth
    "min_samples_split": [2, 5, 10],  # Minimum samples per split
}

# Perform Grid Search for Random Forest
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=3, n_jobs=-1, verbose=1)
rf_grid.fit(X_train_pca, y_train)

# Get best model and parameters
best_rf = rf_grid.best_estimator_
print(f"✅ Best Random Forest Parameters: {rf_grid.best_params_}")

# Predict using best Random Forest
y_pred_rf = best_rf.predict(X_test_pca)

# Evaluate Model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"🌲🌲 Best Random Forest Accuracy: {accuracy_rf:.4f}")
print("\n📊 Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# Save the best Random Forest Model
joblib.dump(best_rf, "best_random_forest.pkl")
print("💾 Best Random Forest Model Saved!")

# ================================
# ✅ Step 4: Compare Model Performance
# ================================

models = ["Decision Tree", "Random Forest"]
accuracies = [accuracy_dt, accuracy_rf]

plt.figure(figsize=(7, 5))
plt.bar(models, accuracies, color=["blue", "green"])
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Decision Tree vs. Random Forest Accuracy")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.show()

# Identify Best Model
best_model = "Decision Tree" if accuracy_dt > accuracy_rf else "Random Forest"
print(f"🏆 Best Model: {best_model}")

# ================================
# ✅ Step 5: Confusion Matrix for Best Model
# ================================

best_y_pred = y_pred_dt if best_model == "Decision Tree" else y_pred_rf
cm_best = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f" Confusion Matrix - {best_model}")
plt.show()
