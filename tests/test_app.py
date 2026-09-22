import io
import os
import json
import pytest
from PIL import Image
import sys
from pathlib import Path

# Add root directory to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import app
from utils.validation import validate_image_file
from utils.prediction import predictor_instance
from database.database import (
    init_db, insert_prediction, get_all_predictions, 
    get_prediction_by_id, delete_prediction
)

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client

def test_health_endpoint(client):
    """Tests the /health endpoint response structure."""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'model_loaded' in data
    assert 'database' in data
    assert data['database'] == 'ok'

def test_image_validation_unsupported_ext():
    """Tests image validation rejecting invalid file extension."""
    class DummyFile:
        filename = "document.pdf"
        def read(self): return b"%PDF-1.4..."
        def seek(self, pos): pass

    is_valid, error, _ = validate_image_file(DummyFile())
    assert is_valid is False
    assert "Unsupported file extension" in error

def test_image_validation_empty_file():
    """Tests image validation rejecting zero-byte files."""
    class DummyFile:
        filename = "leaf.jpg"
        def read(self): return b""
        def seek(self, pos): pass

    is_valid, error, _ = validate_image_file(DummyFile())
    assert is_valid is False
    assert "empty" in error.lower()

def test_image_validation_valid_pil_image():
    """Tests image validation accepting valid PIL RGB image."""
    img = Image.new('RGB', (100, 100), color='green')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()

    class DummyFile:
        filename = "green_leaf.jpg"
        def read(self): return img_bytes
        def seek(self, pos): pass

    is_valid, error, pil_img = validate_image_file(DummyFile())
    assert is_valid is True
    assert error == ""
    assert pil_img is not None
    assert pil_img.size == (100, 100)

def test_class_name_parser():
    """Tests class name parsing logic into Plant Name, Disease Name, and Status."""
    plant, disease, status = predictor_instance.parse_class_name("Tomato___Late_blight")
    assert plant == "Tomato"
    assert disease == "Late Blight"
    assert status == "Diseased"

    plant_h, disease_h, status_h = predictor_instance.parse_class_name("Potato___healthy")
    assert plant_h == "Potato"
    assert disease_h == "Healthy"
    assert status_h == "Healthy"

def test_database_crud():
    """Tests SQLite database insertion, querying, and deletion operations."""
    pred_id = insert_prediction(
        image_name="test_leaf.jpg",
        plant_name="Corn",
        disease_name="Common Rust",
        confidence=95.4,
        status="Diseased",
        image_path="uploads/leaves/test_leaf.jpg"
    )
    assert pred_id is not None

    record = get_prediction_by_id(pred_id)
    assert record is not None
    assert record['plant_name'] == "Corn"
    assert record['confidence'] == 95.4

    all_records = get_all_predictions(search="Corn")
    assert len(all_records) > 0

    deleted = delete_prediction(pred_id)
    assert deleted is True
    assert get_prediction_by_id(pred_id) is None

def test_index_route(client):
    """Tests the home index landing route."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"PhytoVision" in response.data or b"Plant" in response.data


def test_detect_route(client):
    """Tests the /detect route."""
    response = client.get('/detect')
    assert response.status_code == 200
    assert b"Detection" in response.data or b"Leaf" in response.data

def test_dashboard_api(client):
    """Tests the /api/dashboard-stats endpoint."""
    response = client.get('/api/dashboard-stats')
    assert response.status_code == 200
    data = response.get_json()
    assert 'total_predictions' in data
    assert 'healthy_count' in data
    assert 'diseased_count' in data

def test_history_api(client):
    """Tests the /api/history endpoint."""
    response = client.get('/api/history')
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert isinstance(data['records'], list)

def test_encyclopedia_route(client):
    """Tests the /encyclopedia directory route."""
    response = client.get('/encyclopedia')
    assert response.status_code == 200
    assert b"Encyclopedia" in response.data or b"Knowledge" in response.data

def test_about_route(client):
    """Tests the /about architecture route."""
    response = client.get('/about')
    assert response.status_code == 200
    assert b"CBAM" in response.data or b"MobileNetV2" in response.data

def test_predict_no_file(client):
    """Tests /predict returns 400 when no file is submitted."""
    response = client.post('/predict')
    assert response.status_code == 400
    data = response.get_json()
    assert data['success'] is False

