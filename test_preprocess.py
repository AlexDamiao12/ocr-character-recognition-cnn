from PIL import Image, ImageDraw

from api import preprocess_image_for_model


def test_preprocess_keeps_black_on_white_input():
    image = Image.new("L", (280, 280), 255)
    draw = ImageDraw.Draw(image)
    draw.line((110, 30, 110, 250), fill=0, width=18)
    draw.arc((80, 30, 190, 150), start=180, end=360, fill=0, width=18)

    normalized = preprocess_image_for_model(image)

    assert normalized.shape == (1, 28, 28, 1)
    assert float(normalized.mean()) < 0.5
