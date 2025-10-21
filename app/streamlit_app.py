import streamlit as st
import requests
from PIL import Image
import io

st.title("FashionPixel")

# Introducción
st.header("Instrucciones")
st.markdown("""
1. Sube una imagen de prenda de vestir (28x28 px idealmente)
2. La imagen se convertirá a escala de grises
3. Obtendrás la predicción entre 10 categorías
""")

# Carga de imagen
uploaded_file = st.file_uploader("Subir imagen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocesamiento
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    # Mostrar imagen
    st.image(image, caption="Imagen procesada", width=150)

    # Convertir a bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")

    # Enviar a la API
    if st.button("Predecir"):
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": img_byte_arr.getvalue()}
        )

        if response.status_code == 200:
            class_names = [
                "Camiseta", "Pantalón", "Suéter", "Vestido", "Abrigo",
                "Sandalia", "Camisa", "Zapatilla", "Bolso", "Botín"
            ]
            prediction = response.json()["prediction"]
            st.success(f"Predicción: {class_names[prediction]}")
        else:
            st.error("Error en la predicción. Código: {response.status_code}")