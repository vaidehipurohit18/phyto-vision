import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DATABASE_PATH
from database.models import CREATE_PREDICTIONS_TABLE

def get_db_connection():
    """Returns a SQLite database connection with row factory enabled."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes SQLite database tables on application startup."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(CREATE_PREDICTIONS_TABLE)
    conn.commit()
    conn.close()
    print(f"[DATABASE] Initialized SQLite database at: {DATABASE_PATH}")

def insert_prediction(image_name, plant_name, disease_name, confidence, status, image_path, gradcam_path=None):
    """Inserts a new prediction record into SQLite database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (image_name, plant_name, disease_name, confidence, status, image_path, gradcam_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (image_name, plant_name, disease_name, float(confidence), status, image_path, gradcam_path))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id

def get_all_predictions(search=None, status_filter=None):
    """Retrieves all predictions with optional search query and status filter."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if search:
        query += " AND (plant_name LIKE ? OR disease_name LIKE ? OR image_name LIKE ?)"
        pattern = f"%{search}%"
        params.extend([pattern, pattern, pattern])

    if status_filter and status_filter.lower() != 'all':
        query += " AND LOWER(status) = ?"
        params.append(status_filter.lower())

    query += " ORDER BY prediction_date DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_prediction_by_id(prediction_id):
    """Retrieves a single prediction by its primary key ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def delete_prediction(prediction_id):
    """Deletes a prediction record by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0

def get_dashboard_stats():
    """Computes real aggregate metrics and analytics from SQLite database records."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM predictions")
    total = cursor.fetchone()['total']

    if total == 0:
        conn.close()
        return {
            'total_predictions': 0,
            'healthy_count': 0,
            'diseased_count': 0,
            'avg_confidence': 0.0,
            'most_common_disease': 'N/A',
            'status_distribution': {'Healthy': 0, 'Diseased': 0},
            'top_diseases': [],
            'recent_trend': []
        }

    cursor.execute("SELECT COUNT(*) as healthy FROM predictions WHERE LOWER(status) = 'healthy'")
    healthy = cursor.fetchone()['healthy']
    diseased = total - healthy

    cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
    avg_conf = cursor.fetchone()['avg_conf'] or 0.0

    cursor.execute("""
        SELECT disease_name, COUNT(*) as count 
        FROM predictions 
        WHERE LOWER(status) != 'healthy'
        GROUP BY disease_name 
        ORDER BY count DESC 
        LIMIT 1
    """)
    top_row = cursor.fetchone()
    most_common = top_row['disease_name'] if top_row else ('All Healthy' if healthy > 0 else 'N/A')

    # Disease distribution for chart
    cursor.execute("""
        SELECT disease_name, COUNT(*) as count 
        FROM predictions 
        GROUP BY disease_name 
        ORDER BY count DESC 
        LIMIT 6
    """)
    top_diseases = [{'disease': row['disease_name'], 'count': row['count']} for row in cursor.fetchall()]

    # Timeline trend (last 10 records)
    cursor.execute("""
        SELECT DATE(prediction_date) as p_date, COUNT(*) as count, AVG(confidence) as avg_c
        FROM predictions
        GROUP BY DATE(prediction_date)
        ORDER BY p_date ASC
        LIMIT 10
    """)
    trend = [{'date': row['p_date'], 'count': row['count'], 'avg_conf': round(row['avg_c'], 2)} for row in cursor.fetchall()]

    conn.close()

    return {
        'total_predictions': total,
        'healthy_count': healthy,
        'diseased_count': diseased,
        'avg_confidence': round(avg_conf, 2),
        'most_common_disease': most_common,
        'status_distribution': {'Healthy': healthy, 'Diseased': diseased},
        'top_diseases': top_diseases,
        'recent_trend': trend
    }
