import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Read CSV
df = pd.read_csv("train.csv")

# Class names
class_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Display 5 sample images
plt.figure(figsize=(15, 8))

for i in range(5):
    image_id = df.iloc[i]["id_code"]
    diagnosis = df.iloc[i]["diagnosis"]

    image_path = os.path.join("train_images", image_id + ".png")

    image = Image.open(image_path)

    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.title(class_names[diagnosis])
    plt.axis("off")

plt.tight_layout()
plt.show()