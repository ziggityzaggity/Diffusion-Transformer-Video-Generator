import torch
from PIL import Image

from dinov2_encoder import (
    load_dinov2,
    extract_features,
    preprocess
)

# Load pretrained frozen model
model, device = load_dinov2()

print("Device:", device)

# Create a simple sample image or use any test image file 
image = Image.new("RGB", (224, 224), "white")
#image = Image.open("samples/test.png").convert("RGB")

# Preprocess image
x = preprocess(image).unsqueeze(0)

# Extract patch features
features = extract_features(model, x, device)

print("Output shape:", features.shape)

# Check feature dimensions
assert features.shape == (1, 256, 384)

# Confirm that weights are frozen
trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

# Make sure model is frozen
print("Trainable parameters:", trainable)
assert trainable == 0 

print("SUCCESS: DINOv2 works and is frozen")