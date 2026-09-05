import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os


# Custom Dataset
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


# Create dataset
dataset = APTOSDataset(
    "train_split.csv",
    "train_images"
)

# Create DataLoader
loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)


# Test one batch
images, labels = next(iter(loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

print("Labels:", labels)