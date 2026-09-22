CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_name TEXT NOT NULL,
    plant_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT NOT NULL,
    gradcam_path TEXT
);
"""
