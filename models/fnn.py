import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pickle

# 1. Definición de la clase SimpleFNN (asegúrate de que coincida con la que usaste para entrenar)
class SimpleFNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. Hiperparámetros (deben coincidir con los de tu modelo entrenado)
input_size = 784
hidden_size = 512  
num_classes = 10
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 3. Cargar el modelo
@st.cache_resource
def load_model(path):
    model = SimpleFNN(input_size, hidden_size, num_classes)
    with open(path, 'rb') as f:
        model.load_state_dict(pickle.load(f))
    model.eval()
    return model

model = load_model('best_fnn_model.pkl')