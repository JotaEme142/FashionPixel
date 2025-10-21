import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, input_channels, L, conv_filters, activation_fn, num_classes=10):
        super(CustomCNN, self).__init__()
        layers = []
        in_channels = input_channels

        for i in range(L):  # Repite 'L' veces para agregar capas convolucionales.
            layers.append(nn.Conv2d(in_channels, conv_filters[i], kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(conv_filters[i]))  # Normalización por lotes
            layers.append(activation_fn)  # Función de activación (ReLU, LeakyReLU, etc.)
            in_channels = conv_filters[i]  # Actualiza el número de canales de entrada.

        # Combinación de todas las capas convolucionales en una secuencia.
        self.conv_layers = nn.Sequential(*layers)

        # Simulación de una entrada para calcular el tamaño de salida de las capas convolucionales
        dummy_input = torch.randn(1, 1, 28, 28)  # Imagen de entrada simulada (MNIST)
        with torch.no_grad():  # No queremos calcular gradientes aquí
            dummy_output = self.conv_layers(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]  # Obtiene el tamaño dinámico

        # Ahora usa 'flattened_size' para la capa completamente conectada
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, num_classes)  # Tamaño dinámico en lugar de un valor fijo
        )

        # Calcular el número de características de salida de las capas convolucionales
        self.num_features = conv_filters[-1] * dummy_output.size(2) * dummy_output.size(3)  # Tamaño final después de convoluciones

        # Definir la capa de clasificación como una red completamente conectada
        self.fc1 = nn.Linear(self.num_features, 128)  # Primera capa oculta
        self.fc2 = nn.Linear(128, 64)  # Segunda capa oculta
        self.fc3 = nn.Linear(64, 32)  # Tercera capa oculta
        self.fc4 = nn.Linear(32, num_classes)  # Capa de salida

        # Función de activación
        self.activation_fn = activation_fn

    def forward(self, x):
        # Propagación hacia adelante: pasa los datos por las capas convolucionales.
        x = self.conv_layers(x)

        # Aplanar el tensor para la capa completamente conectada.
        x = x.view(x.size(0), -1)  # Aplana el tensor a [batch_size, num_features]

        # Pasar por las capas completamente conectadas
        x = self.activation_fn(self.fc1(x))  # Primera capa oculta
        x = self.activation_fn(self.fc2(x))  # Segunda capa oculta
        x = self.activation_fn(self.fc3(x))  # Tercera capa oculta
        x = self.fc4(x)  # Capa de salida (sin activación)

        return x