import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Environment settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'plant-disease-cbam-attention-secret-key-2026')
DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')

# Directory Paths
MODEL_DIR = BASE_DIR / 'model'
DATASET_DIR = BASE_DIR / 'dataset'
TRAINING_DIR = BASE_DIR / 'training'
RESULTS_DIR = TRAINING_DIR / 'results'
DATA_DIR = BASE_DIR / 'data'
DATABASE_DIR = BASE_DIR / 'database'
UPLOAD_FOLDER = BASE_DIR / 'uploads'

# Ensure directories exist
for folder in [MODEL_DIR, UPLOAD_FOLDER, RESULTS_DIR, DATA_DIR, DATABASE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# File Paths
MODEL_PATH = BASE_DIR / "model" / "plant_disease_finetuned.keras"
CLASS_NAMES_PATH = MODEL_DIR / 'class_names.json'
MODEL_METADATA_PATH = MODEL_DIR / 'model_metadata.json'
DATABASE_PATH = DATABASE_DIR / 'plant_disease.db'
DISEASE_INFO_PATH = DATA_DIR / 'disease_info.json'

# Model & Prediction Configuration
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
CONFIDENCE_THRESHOLD = 0.60

# File Upload Configuration
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
