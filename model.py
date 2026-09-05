import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# Load pretrained EfficientNet-B0
weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(weights=weights)


# Get the number of inputs to the final layer
num_features = model.classifier[1].in_features


# Replace the final layer
model.classifier[1] = nn.Linear(
    num_features,
    5
)


# Print the final layer
print("Final classification layer:")
print(model.classifier[1])

print("\nNumber of output classes:", 5)