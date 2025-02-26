import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet18_Weights
from PIL import Image
import os

# ✅ Ensure multiprocessing for Windows
if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    # ✅ Use GPU if Available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Using device: {device}")

    # ✅ Load dataset
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    # ✅ Ensure correct shape
    if X_train.ndim == 2:
        num_samples = X_train.shape[0]
        try:
            X_train = X_train.reshape(num_samples, 224, 224, 3)
            X_test = X_test.reshape(X_test.shape[0], 224, 224, 3)
        except ValueError:
            raise ValueError(f"❌ Cannot reshape X_train with shape {X_train.shape} into (N, 224, 224, 3). Check dataset.")

    # ✅ Dataset Class with Stronger Augmentations
    class TumorDataset(Dataset):
        def __init__(self, X, y, train=True):
            self.X = [Image.fromarray((img * 255).astype(np.uint8)) for img in X]  # Convert NumPy array to PIL
            self.y = torch.tensor(y, dtype=torch.long)

            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomAffine(0, shear=20, scale=(0.6, 1.3)),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]) if train else transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            img = self.transform(self.X[idx])  # Apply transforms to PIL image
            return img, self.y[idx]

    # ✅ Create DataLoaders (Adjust `num_workers` for CPU)
    num_workers = 2 if torch.cuda.is_available() else 0  # Fix multiprocessing issue on Windows
    train_dataset = TumorDataset(X_train, y_train, train=True)
    test_dataset = TumorDataset(X_test, y_test, train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)

    # ✅ Load Pretrained ResNet18 with More Regularization
    class ModifiedResNet18(nn.Module):
        def __init__(self):
            super(ModifiedResNet18, self).__init__()
            self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),  # ✅ Increased Dropout to 0.5
                nn.Linear(self.model.fc.in_features, 3)
            )

        def forward(self, x):
            return self.model(x)

    model = ModifiedResNet18().to(device)

    # ✅ Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # ✅ Added Label Smoothing
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-3)  # ✅ Increased Regularization
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # ✅ Early Stopping Parameters
    early_stopping_patience = 5
    best_acc = 0.0
    epochs_no_improve = 0

    # ✅ Track Metrics
    train_losses, test_losses, train_acc, test_acc = [], [], [], []

    # ✅ Training Loop
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ✅ Gradient Clipping
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_losses.append(running_loss / len(train_loader))
        train_acc.append(correct / total)

        # ✅ Evaluate Model
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        test_losses.append(test_loss / len(test_loader))
        test_acc.append(correct / total)

        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {train_losses[-1]:.4f} - Train Acc: {train_acc[-1]:.4f} - Test Acc: {test_acc[-1]:.4f}")

        scheduler.step()

        # ✅ Early Stopping
        if test_acc[-1] > best_acc:
            best_acc = test_acc[-1]
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")  # ✅ Save Best Model
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("⏹️ Early stopping triggered. Stopping training.")
                break

    # ✅ Load Best Model
    model.load_state_dict(torch.load("best_model.pth"))
    print("\n🎯 Training Completed Successfully!")

