import io
import os

import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
api_key = os.getenv("API_KEY", "")
min_confidence = float(os.getenv("MIN_CONFIDENCE", "0.85"))

st.set_page_config(page_title="CNN Character Recognition", layout="centered")
st.title("CNN Character Recognition")

canvas_result = None
uploaded_file = None

option = st.radio("Choose an input method:", ["Upload image", "Draw"], horizontal=True)
st.caption("Tip: use one clear character, in black on a white background for best results.")

if option == "Upload image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
else:
    st.write("Draw a character:")
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=8,
        stroke_color="black",
        background_color="white",
        width=280,
        height=280,
        drawing_mode="freedraw",
        return_image_data=True,
        update_streamlit=True,
        key="canvas",
    )

with st.form("prediction_form"):
    submitted = st.form_submit_button("Recognize character")

img = None

if option == "Upload image" and uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
elif option == "Draw" and canvas_result is not None and canvas_result.image_data is not None:
    img = Image.fromarray(np.uint8(canvas_result.image_data)).convert("L")

if submitted and img is not None:
    st.image(img, caption="Input image", width=150)

    if not api_key:
        st.error("Set the API_KEY environment variable to use the API.")
    else:
        image_buffer = io.BytesIO()
        img.save(image_buffer, format="PNG")
        try:
            response = requests.post(
                f"{api_url}/predict",
                headers={"X-API-Key": api_key},
                files={"file": ("character.png", image_buffer.getvalue(), "image/png")},
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            confidence = float(result.get("confidence", 0.0))

            if confidence < min_confidence:
                st.warning(
                    f"Low confidence prediction: {result.get('character', '?')} "
                    f"({confidence:.1%}). Please draw a cleaner character or upload a sharper image."
                )
            else:
                st.success(
                    f"Recognized character: {result['character']} "
                    f"({confidence:.1%})"
                )
        except requests.RequestException as error:
            st.error(f"Could not contact the API: {error}")

elif submitted:
    st.warning("Please upload an image or draw a character before submitting.")
