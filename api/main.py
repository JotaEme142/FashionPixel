from fastapi import FastAPI, File, UploadFile
import torch
from torch import nn
from PIL import Image
from model import CustomCNN
from torchvision import transforms
import numpy as np
import io

app = FastAPI(title="Clasificador Fashion MNIST", version="1.0")

# Cargar el modelo con los parámetros óptimos
model_params = {
    "input_channels": 1,
    "L": 4,
    "conv_filters": [512, 256, 128, 64],
    "activation_fn": nn.ReLU(),
    "num_classes": 10
}

with open('model.pkl', 'rb') as file:
    model = torch.load(file, map_location=torch.device('cpu'),weights_only=False)

model.eval()  # Poner el modelo en modo de evaluación

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Procesar imagen
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Transformaciones idénticas al entrenamiento
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(image).unsqueeze(0)  # Añadir dimensión batch

    # Inferencia
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output).item()

    return {"prediction": prediction}