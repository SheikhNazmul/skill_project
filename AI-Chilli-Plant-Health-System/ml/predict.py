"""Predict the chilli leaf disease class for one image."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def load_model(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu")
    classes = payload["classes"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("ml/artifacts/best_model.pt"))
    args = parser.parse_args()

    model, classes = load_model(args.model)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(image), dim=1)[0]
        index = int(probabilities.argmax())

    print(f"Prediction: {classes[index]}")
    print(f"Confidence: {probabilities[index].item():.2%}")


if __name__ == "__main__":
    main()
