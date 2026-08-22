"""Simple Streamlit UI for the trained chilli leaf classifier."""

from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

MODEL_PATH = Path(__file__).parent / "artifacts" / "best_model.pt"

st.set_page_config(page_title="Chilli Plant Health AI", page_icon="🌶️")
st.title("🌶️ Chilli Plant Disease Detector")
st.caption("University final-semester AI module")

@st.cache_resource
def load_model():
    payload = torch.load(MODEL_PATH, map_location="cpu")
    classes = payload["classes"]
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, classes

uploaded = st.file_uploader("Upload a chilli leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded leaf", use_container_width=True)

    if not MODEL_PATH.exists():
        st.info("No trained checkpoint is included yet. Run train.py first to create ml/artifacts/best_model.pt.")
    else:
        model, classes = load_model()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with torch.no_grad():
            probabilities = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0]
        index = int(probabilities.argmax())
        st.success(f"Prediction: {classes[index]}")
        st.metric("Confidence", f"{probabilities[index].item():.2%}")
