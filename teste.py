import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Carregar modelo e mapeamento
model = load_model("modelo_emnist_balanced.keras")
label_map = np.load("label_map.npy", allow_pickle=True).item()
emnist_classes = [chr(label_map[k]) for k in sorted(label_map.keys())]

st.title("CNN-Reconhecimento de Caracteres")

# Opção de entrada: upload ou desenho
option = st.radio("Escolhe uma forma de entrada:", ["Upload de imagem", "Desenhar"])

img_array = None

if option == "Upload de imagem":
    uploaded_file = st.file_uploader("Faz upload de uma imagem", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('L')
        img = img.resize((28, 28))  # Sem rotação
        img_array = np.array(img)

elif option == "Desenhar":
    st.write("Desenha um caracter:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=8,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        img = Image.fromarray(np.uint8(canvas_result.image_data)).convert('L')
        img = ImageOps.invert(img)  # Inverter fundo branco para preto
        img = img.resize((28, 28))  # Sem rotação
        img_array = np.array(img)

# Se houver imagem (upload ou desenho), processa
if img_array is not None:
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    class_index = int(np.argmax(prediction))
    recognized_char = emnist_classes[class_index]

    st.image(img, caption="Imagem usada", width=150)
    st.success(f"Caracter reconhecido: {recognized_char}")
