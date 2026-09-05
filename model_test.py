import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


# -----------------------------
# Dataset
# -----------------------------

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


# -----------------------------
# Load one batch
# -----------------------------

dataset = APTOSDataset(
    "train_split.csv",
    "train_images"
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

images, labels = next(iter(loader))


# -----------------------------
# Load EfficientNet
# -----------------------------

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    5
)


# -----------------------------
# Test prediction
# -----------------------------

model.eval()

with torch.no_grad():
    outputs = model(images)

predictions = torch.argmax(outputs, dim=1)


print("Input shape:")
print(images.shape)

print("\nOutput shape:")
print(outputs.shape)

print("\nActual labels:")
print(labels)

print("\nPredicted labels:")
print(predictions)