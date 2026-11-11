import torch
from torchvision import transforms

# Using DINOv2 (ViT-S/14) (can update to something else if we would like later)
MODEL_NAME = "dinov2_vits14"

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_dinov2(device=None):
    """Download, load and freeze DINOv2."""
    # try to use gpu if available, otherwise use cpu
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu" 

    model = torch.hub.load(
        "facebookresearch/dinov2",
        MODEL_NAME
    )

    # Freeze the model's pretrained weights
    model.requires_grad_(False)
    model.eval()
    model = model.to(device)

    return model, device


def extract_features(model, frames, device):
    """
    Input shape: [Batch_size, 3, 224, 224]
    Output shape: [Batch_size, 256, 384]
    """
    with torch.inference_mode():
        frames = frames.to(device)
        result = model.forward_features(frames)
        patches = result["x_norm_patchtokens"]

    return patches.cpu()