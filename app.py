import os
import json
import uuid
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY, DEBUG, UPLOAD_FOLDER, DISEASE_INFO_PATH, 
    CLASS_NAMES_PATH, MODEL_PATH, DATABASE_PATH, CONFIDENCE_THRESHOLD
)
from database.database import (
    init_db, insert_prediction, get_all_predictions, 
    get_prediction_by_id, delete_prediction, get_dashboard_stats
)
from utils.validation import validate_image_file
from utils.prediction import predictor_instance
from utils.gradcam import generate_gradcam_heatmap, save_gradcam_overlay

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Initialize SQLite Database on server start
with app.app_context():
    init_db()

def load_disease_info():
    """Helper to load disease knowledge base JSON."""
    if not DISEASE_INFO_PATH.exists():
        return []
    with open(DISEASE_INFO_PATH, 'r') as f:
        return json.load(f)

def get_supported_class_names():
    """Helper to load list of class names supported by the trained AI model."""
    if not CLASS_NAMES_PATH.exists():
        return []
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            classes_dict = json.load(f)
            return list(classes_dict.values())
    except Exception:
        return []

# --------------------------------------------------------------------------
# ROUTES
# --------------------------------------------------------------------------

@app.route('/')
def index():
    """Home landing page."""
    supported_classes = get_supported_class_names()
    model_loaded = predictor_instance.is_loaded
    return render_template('index.html', model_loaded=model_loaded, supported_count=len(supported_classes))

@app.route('/detect')
def detect():
    """Leaf image detection upload interface."""
    model_loaded = predictor_instance.is_loaded
    return render_template('detect.html', model_loaded=model_loaded)
@app.route('/predict', methods=['POST'])
def predict():
    """
    Real AI prediction endpoint with detailed Render diagnostics.

    This version logs every major stage so we can identify
    exactly where a production timeout/502 occurs.
    """

    start_time = time.time()

    def log(message):
        elapsed = time.time() - start_time
        print(
            f"[PREDICT +{elapsed:.2f}s] {message}",
            flush=True
        )

    log("========== PREDICTION REQUEST STARTED ==========")

    # ---------------------------------------------------------
    # CHECK MODEL
    # ---------------------------------------------------------

    log("Checking model status...")

    if not predictor_instance.is_loaded:
        log("Model is NOT loaded. Loading model now...")

        predictor_instance.load_model_and_classes()

        log(
            f"Model loading finished. Loaded={predictor_instance.is_loaded}"
        )

        if not predictor_instance.is_loaded:
            log("ERROR: Model could not be loaded.")

            return jsonify({
                'success': False,
                'error': 'AI model could not be loaded.'
            }), 400

    log("Model is loaded and ready.")

    # ---------------------------------------------------------
    # CHECK IMAGE
    # ---------------------------------------------------------

    log("Checking uploaded image...")

    if 'leaf_image' not in request.files:
        log("ERROR: No leaf_image in request.files.")

        return jsonify({
            'success': False,
            'error': 'No image file uploaded.'
        }), 400

    file = request.files['leaf_image']

    log(
        f"Received file: {file.filename} "
        f"content_type={file.content_type}"
    )

    # ---------------------------------------------------------
    # VALIDATE IMAGE
    # ---------------------------------------------------------

    log("Starting image validation...")

    is_valid, error_msg, pil_image = validate_image_file(file)

    log(
        f"Image validation completed. "
        f"valid={is_valid}"
    )

    if not is_valid:
        log(f"Image validation failed: {error_msg}")

        return jsonify({
            'success': False,
            'error': error_msg
        }), 400

    try:

        # -----------------------------------------------------
        # SAVE ORIGINAL IMAGE
        # -----------------------------------------------------

        log("Starting original image save...")

        unique_id = str(uuid.uuid4())[:8]

        safe_filename = secure_filename(
            file.filename
        )

        filename = (
            f"{unique_id}_{safe_filename}"
        )

        leaf_upload_dir = (
            UPLOAD_FOLDER / 'leaves'
        )

        leaf_upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        saved_image_path = (
            leaf_upload_dir / filename
        )

        pil_image.save(
            saved_image_path
        )

        rel_image_path = (
            f"uploads/leaves/{filename}"
        )

        log(
            f"Original image saved: {saved_image_path}"
        )

        # -----------------------------------------------------
        # REAL MODEL PREDICTION
        # -----------------------------------------------------

        log("========== STARTING MODEL PREDICTION ==========")

        prediction_start = time.time()

        pred_result = (
            predictor_instance.predict(
                pil_image
            )
        )

        prediction_time = time.time() - prediction_start

        log(
            f"MODEL PREDICTION FINISHED "
            f"in {prediction_time:.2f}s"
        )

        log(
            f"Prediction result: {pred_result}"
        )

        # -----------------------------------------------------
        # EXTRACT PREDICTION
        # -----------------------------------------------------

        plant_name = pred_result.get(
            'plant_name',
            'Unknown Plant'
        )

        disease_name = pred_result.get(
            'disease_name',
            'Unknown Condition'
        )

        status = pred_result.get(
            'status',
            'Unknown'
        )

        confidence = pred_result.get(
            'confidence',
            0
        )

        log(
            f"Plant={plant_name} | "
            f"Disease={disease_name} | "
            f"Status={status} | "
            f"Confidence={confidence}"
        )

        # -----------------------------------------------------
        # GRAD-CAM
        # -----------------------------------------------------

        gradcam_rel_path = None

        log("========== STARTING GRAD-CAM ==========")

        gradcam_start = time.time()

        if predictor_instance.model is not None:

            log("Model object exists. Finding predicted class...")

            top_class_name = (
                pred_result.get(
                    'raw_class'
                )
            )

            log(
                f"Predicted raw class: {top_class_name}"
            )

            class_idx = None

            for idx, c_name in (
                predictor_instance
                .class_names
                .items()
            ):

                if c_name == top_class_name:

                    class_idx = idx

                    break

            log(
                f"Grad-CAM class index: {class_idx}"
            )

            if class_idx is not None:

                try:

                    log(
                        "Importing preprocessing function..."
                    )

                    from utils.preprocessing import (
                        preprocess_leaf_image
                    )

                    log(
                        "Starting image preprocessing for Grad-CAM..."
                    )

                    batch_tensor = (
                        preprocess_leaf_image(
                            pil_image
                        )
                    )

                    log(
                        "Grad-CAM preprocessing finished."
                    )

                    log(
                        "Calling generate_gradcam_heatmap()..."
                    )

                    heatmap_start = time.time()

                    heatmap = (
                        generate_gradcam_heatmap(
                            predictor_instance.model,
                            batch_tensor,
                            class_idx
                        )
                    )

                    heatmap_time = time.time() - heatmap_start

                    log(
                        f"generate_gradcam_heatmap() "
                        f"finished in {heatmap_time:.2f}s"
                    )

                    if heatmap is not None:

                        log(
                            "Heatmap generated successfully."
                        )

                        log(
                            "Saving Grad-CAM overlay..."
                        )

                        gradcam_rel_path = (
                            save_gradcam_overlay(
                                pil_image,
                                heatmap,
                                f"gradcam_{filename}"
                            )
                        )

                        log(
                            f"Grad-CAM overlay saved: "
                            f"{gradcam_rel_path}"
                        )

                    else:

                        log(
                            "WARNING: Grad-CAM returned None."
                        )

                except Exception as gradcam_error:

                    log(
                        f"WARNING: Grad-CAM failed: "
                        f"{gradcam_error}"
                    )

                    app.logger.exception(
                        "Grad-CAM exception"
                    )

            else:

                log(
                    "WARNING: Could not find class index "
                    "for Grad-CAM."
                )

        else:

            log(
                "WARNING: predictor_instance.model is None."
            )

        gradcam_time = time.time() - gradcam_start

        log(
            f"TOTAL GRAD-CAM STAGE TIME: "
            f"{gradcam_time:.2f}s"
        )

        # -----------------------------------------------------
        # SAVE TO DATABASE
        # -----------------------------------------------------

        log("========== STARTING DATABASE SAVE ==========")

        database_start = time.time()

        prediction_id = insert_prediction(

            image_name=safe_filename,

            plant_name=plant_name,

            disease_name=disease_name,

            confidence=confidence,

            status=status,

            image_path=rel_image_path,

            gradcam_path=gradcam_rel_path

        )

        database_time = time.time() - database_start

        log(
            f"Database save finished in "
            f"{database_time:.2f}s"
        )

        log(
            f"Prediction ID: {prediction_id}"
        )

        # -----------------------------------------------------
        # RESPONSE
        # -----------------------------------------------------

        total_time = time.time() - start_time

        log(
            "========== PREDICTION COMPLETE =========="
        )

        log(
            f"TOTAL REQUEST TIME: {total_time:.2f}s"
        )

        return jsonify({

            'success': True,

            'prediction_id': prediction_id,

            'redirect_url': url_for(
                'result',
                id=prediction_id
            )

        })

    except RuntimeError as re:

        total_time = time.time() - start_time

        log(
            f"RUNTIME ERROR after {total_time:.2f}s: "
            f"{re}"
        )

        app.logger.exception(
            "Prediction RuntimeError"
        )

        return jsonify({
            'success': False,
            'error': str(re)
        }), 400

    except Exception as e:

        total_time = time.time() - start_time

        log(
            f"FATAL PREDICTION ERROR after "
            f"{total_time:.2f}s: {e}"
        )

        app.logger.exception(
            "Prediction pipeline failed"
        )

        return jsonify({

            'success': False,

            'error':
                f"Inference pipeline error: {str(e)}"

        }), 500

@app.route('/result/<int:id>')
def result(id):
    """Prediction result view showing disease metrics, Grad-CAM, and encyclopedia linkage."""
    record = get_prediction_by_id(id)
    if not record:
        return render_template('404.html', message="Prediction record not found."), 404

    diseases = load_disease_info()
    
    # Match disease info
    matching_info = None
    plant_lower = record['plant_name'].lower()
    disease_lower = record['disease_name'].lower()

    for item in diseases:
        if item['plant'].lower() in plant_lower or plant_lower in item['plant'].lower():
            if item['disease'].lower() in disease_lower or disease_lower in item['disease'].lower():
                matching_info = item
                break
    
    # Fallback to general plant match if disease mismatch
    if not matching_info:
        for item in diseases:
            if item['plant'].lower() in plant_lower:
                matching_info = item
                break

    is_low_confidence = record['confidence'] < (CONFIDENCE_THRESHOLD * 100.0)

    return render_template(
        'result.html',
        record=record,
        disease_info=matching_info,
        is_low_confidence=is_low_confidence,
        confidence_threshold_pct=int(CONFIDENCE_THRESHOLD * 100.0)
    )

@app.route('/dashboard')
def dashboard():
    """Analytics Dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/dashboard-stats')
def api_dashboard_stats():
    """API endpoint delivering dynamic SQLite statistics for Chart.js."""
    stats = get_dashboard_stats()
    return jsonify(stats)

@app.route('/history')
def history():
    """Prediction History page."""
    return render_template('history.html')

@app.route('/api/history', methods=['GET'])
def api_history():
    """API endpoint retrieving filtered prediction records from SQLite."""
    search = request.args.get('search', '').strip()
    status_filter = request.args.get('status', 'all').strip()
    records = get_all_predictions(search=search, status_filter=status_filter)
    return jsonify({'success': True, 'records': records})

@app.route('/api/history/<int:id>', methods=['DELETE'])
def api_delete_history(id):
    """API endpoint deleting a prediction record from SQLite."""
    deleted = delete_prediction(id)
    if deleted:
        return jsonify({'success': True, 'message': 'Prediction record deleted successfully.'})
    return jsonify({'success': False, 'error': 'Record not found or could not be deleted.'}), 404

@app.route('/encyclopedia')
def encyclopedia():
    """Plant Disease Knowledge Base Directory."""
    diseases = load_disease_info()
    supported_classes = get_supported_class_names()

    # Determine unique plants for filter
    plants = sorted(list(set(item['plant'] for item in diseases)))

    # Mark whether each disease is supported by current AI model
    for item in diseases:
        item_plant = item['plant'].lower()
        item_disease = item['disease'].lower()
        item['is_supported'] = False
        
        for sc in supported_classes:
            sc_lower = sc.lower().replace('_', ' ')
            if item_plant in sc_lower and (item_disease in sc_lower or 'healthy' in item_disease):
                item['is_supported'] = True
                break

    return render_template('encyclopedia.html', diseases=diseases, plants=plants)

@app.route('/encyclopedia/<disease_id>')
def encyclopedia_detail(disease_id):
    """Detailed view for a specific plant disease entry."""
    diseases = load_disease_info()
    disease_entry = next((item for item in diseases if item['id'] == disease_id), None)
    
    if not disease_entry:
        return render_template('404.html', message="Disease entry not found in encyclopedia."), 404

    supported_classes = get_supported_class_names()
    is_supported = False
    for sc in supported_classes:
        sc_lower = sc.lower().replace('_', ' ')
        if disease_entry['plant'].lower() in sc_lower and disease_entry['disease'].lower() in sc_lower:
            is_supported = True
            break

    return render_template('disease.html', disease=disease_entry, is_supported=is_supported)

@app.route('/about')
def about():
    """About page detailing CBAM architecture, model specifications, and viva Q&A."""
    model_loaded = predictor_instance.is_loaded
    metadata = {}
    if MODEL_PATH.parent.joinpath('model_metadata.json').exists():
        try:
            with open(MODEL_PATH.parent.joinpath('model_metadata.json'), 'r') as f:
                metadata = json.load(f)
        except Exception:
            pass

    return render_template('about.html', model_loaded=model_loaded, metadata=metadata)

@app.route('/health')
def health():
    """Health check endpoint returning dynamic system status."""
    db_ok = DATABASE_PATH.exists()
    model_ok = predictor_instance.is_loaded
    
    status_str = "ok" if (db_ok and model_ok) else "degraded"
    
    return jsonify({
        'status': status_str,
        'model_loaded': model_ok,
        'database': 'ok' if db_ok else 'missing',
        'confidence_threshold': CONFIDENCE_THRESHOLD
    })

# Serve uploaded static media files safely
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# PWA Service Worker and Manifest routes
@app.route('/manifest.json')
def manifest():
    return send_from_directory('.', 'manifest.json')

@app.route('/service-worker.js')
def service_worker():
    return send_from_directory('.', 'service-worker.js')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', message="The requested page could not be found."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('404.html', message="An unexpected server error occurred."), 500

if __name__ == '__main__':
    print("=" * 60)
    print("PLANT DISEASE DETECTION & CLASSIFICATION FLASK APP")
    print(f"Model Status: {'LOADED & READY' if predictor_instance.is_loaded else 'NOT TRAINED YET'}")
    print("Running on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=DEBUG)
