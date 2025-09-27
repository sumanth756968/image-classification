# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import models
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    resnet50, ResNet50_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

# -----------------------------
# Load models with latest weights
# -----------------------------
@st.cache_resource
def load_models():
    return {
        "MobileNetV2": (mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT), MobileNet_V2_Weights.DEFAULT),
        "ResNet50": (resnet50(weights=ResNet50_Weights.DEFAULT), ResNet50_Weights.DEFAULT),
        "EfficientNetB0": (efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT), EfficientNet_B0_Weights.DEFAULT)
    }

models_dict = load_models()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🤖 Multi-Model AI Image Classifier")
st.write("Upload an image and choose a model to classify it. Models are pretrained on ImageNet.")

# Model selection
model_name = st.selectbox("Select Model", list(models_dict.keys()))
model, weights = models_dict[model_name]
model.eval()  # evaluation mode
preprocess = weights.transforms()

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess
    input_tensor = preprocess(image).unsqueeze(0)
    
    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        class_name = weights.meta["categories"][predicted_idx.item()]
    
    st.success(f"Predicted Class ({model_name}): **{class_name}**")
