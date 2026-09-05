import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import os


# ==========================================
# 1. Dataset
# ==========================================

class APTOSDataset(Dataset):

    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image_id = self.data.iloc[index]["id_code"]
        label = self.data.iloc[index]["diagnosis"]

        image_path = os.path.join(
            self.image_dir,
            image_id + ".png"
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        return image, label


# ==========================================
# 2. Load datasets
# ==========================================

train_dataset = APTOSDataset(
    "train_split.csv",
    "train_images"
)

val_dataset = APTOSDataset(
    "val_split.csv",
    "train_images"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False
)


# ==========================================
# 3. Device
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ==========================================
# 4. Load EfficientNet-B0
# ==========================================

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    5
)

model = model.to(device)


# ==========================================
# 5. Class weights
# ==========================================

class_counts = torch.tensor(
    [1444, 296, 799, 154, 236],
    dtype=torch.float
)

class_weights = 1.0 / class_counts

class_weights = class_weights / class_weights.sum() * 5

class_weights = class_weights.to(device)

print("Class weights:")
print(class_weights)


# ==========================================
# 6. Loss and optimizer
# ==========================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# ==========================================
# 7. Training — 1 epoch
# ==========================================

epochs = 1

for epoch in range(epochs):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    print("\nTraining...")

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model
        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        # Print progress
        if (batch_index + 1) % 20 == 0:
            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"Loss: {loss.item():.4f}"
            )

    train_accuracy = 100 * correct / total

    print("\nTraining Results")
    print("----------------")
    print("Loss:", running_loss / len(train_loader))
    print("Accuracy:", train_accuracy)


# ==========================================
# 8. Validation
# ==========================================

model.eval()

correct = 0
total = 0

print("\nValidation...")

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()


val_accuracy = 100 * correct / total

print("\nValidation Accuracy:", val_accuracy)


# ==========================================
# 9. Save model
# ==========================================

torch.save(
    model.state_dict(),
    "efficientnet_dr_model.pth"
)

print("\nModel saved successfully!")