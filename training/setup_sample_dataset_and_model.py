import os
import json
import time
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import sys

# Add root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR, MODEL_PATH, CLASS_NAMES_PATH, 
    MODEL_METADATA_PATH, IMAGE_SIZE, RESULTS_DIR
)

# Standard PlantVillage 38 Class Names
PLANT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def generate_synthetic_leaf(class_name, size=(224, 224)):
    """Generates a synthetic leaf image with class-dependent visual characteristics."""
    # Base green leaf canvas
    img = Image.new("RGB", size, (34, 112, 58))
    draw = ImageDraw.Draw(img)

    # Leaf vein structure
    draw.line([(112, 10), (112, 214)], fill=(45, 140, 70), width=4)
    draw.line([(112, 60), (50, 100)], fill=(40, 130, 65), width=2)
    draw.line([(112, 60), (174, 100)], fill=(40, 130, 65), width=2)
    draw.line([(112, 120), (40, 160)], fill=(40, 130, 65), width=2)
    draw.line([(112, 120), (184, 160)], fill=(40, 130, 65), width=2)

    # Add disease spots if diseased
    if "healthy" not in class_name.lower():
        num_spots = np.random.randint(5, 15)
        for _ in range(num_spots):
            x = np.random.randint(20, 204)
            y = np.random.randint(20, 204)
            r = np.random.randint(5, 20)
            
            if "rust" in class_name.lower() or "orange" in class_name.lower():
                color = (200, 90, 20) # Orange rust
            elif "blight" in class_name.lower() or "rot" in class_name.lower() or "spot" in class_name.lower():
                color = (80, 50, 20) # Dark brown spot
            elif "powdery" in class_name.lower() or "mold" in class_name.lower():
                color = (220, 220, 210) # White powdery spot
            else:
                color = (130, 120, 30) # Yellowish spot

            draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    return img

def setup_synthetic_dataset(images_per_class=3):
    """Creates train, validation, and test dataset folders with synthetic leaf images."""
    print("Generating synthetic dataset structure...")
    for split in ['train', 'validation', 'test']:
        split_dir = DATASET_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for c_idx, c_name in enumerate(PLANT_CLASSES):
            c_dir = split_dir / c_name
            c_dir.mkdir(parents=True, exist_ok=True)
            
            count = images_per_class if split == 'train' else 1
            for i in range(count):
                img_path = c_dir / f"leaf_{i+1}.jpg"
                if not img_path.exists():
                    img = generate_synthetic_leaf(c_name)
                    img.save(img_path, quality=90)

    print(f"[SUCCESS] Synthetic dataset created in {DATASET_DIR}")

if __name__ == "__main__":
    setup_synthetic_dataset()
