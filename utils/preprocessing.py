import numpy as np
from PIL import Image
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from config import IMAGE_SIZE


def preprocess_leaf_image(pil_image):
    """
    Preprocess image for inference.

    IMPORTANT:
    MobileNetV2 preprocessing is already included INSIDE
    model_architecture.py.

    Therefore this function must ONLY:
        1. Convert image to RGB
        2. Resize image
        3. Convert to float32
        4. Add batch dimension

    DO NOT apply mobilenet_v2.preprocess_input() here.
    """

    # Ensure RGB image
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    # Resize image
    resized_img = pil_image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    # Convert to NumPy float32
    img_array = np.array(
        resized_img,
        dtype=np.float32
    )

    # Add batch dimension
    batch_array = np.expand_dims(
        img_array,
        axis=0
    )

    return batch_array