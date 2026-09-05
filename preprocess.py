import cv2
import matplotlib.pyplot as plt

# Image path
image_path = "train_images/000c1434d8d7.png"

# Read image
image = cv2.imread(image_path)

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize
resized = cv2.resize(image_rgb, (224, 224))

# Convert to LAB for contrast enhancement
lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)

# Split LAB channels
l, a, b = cv2.split(lab)

# CLAHE contrast enhancement
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

l = clahe.apply(l)

# Merge channels
enhanced_lab = cv2.merge((l, a, b))

# Convert back to RGB
enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

# Normalize pixel values
normalized = enhanced / 255.0

# Display
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(enhanced)
plt.title("Contrast Enhanced")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(normalized)
plt.title("Normalized")
plt.axis("off")

plt.tight_layout()
plt.show()