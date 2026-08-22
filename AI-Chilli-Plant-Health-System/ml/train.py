"""Train a chilli leaf disease classifier with transfer learning."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("ml/artifacts/best_model.pt"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    if len(dataset) < 2:
        raise ValueError("Dataset must contain at least two images.")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    model = build_model(len(dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_accuracy = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device))
                predictions = outputs.argmax(dim=1).cpu()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total else 0.0
        print(f"Epoch {epoch + 1}/{args.epochs} - validation accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "classes": dataset.classes,
        "image_size": 224,
    }, args.output)
    print(f"Saved best model to {args.output} (accuracy={best_accuracy:.4f})")


if __name__ == "__main__":
    main()
