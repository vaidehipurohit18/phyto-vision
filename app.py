import os

# Keep CPU usage low on Render
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import uuid
import time
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    url_for,
    send_from_directory
)

from werkzeug.utils import secure_filename

from config import (
    SECRET_KEY,
    DEBUG,
    UPLOAD_FOLDER,
    DISEASE_INFO_PATH,
    CLASS_NAMES_PATH,
    MODEL_PATH,
    DATABASE_PATH,
    CONFIDENCE_THRESHOLD
)

from database.database import (
    init_db,
    insert_prediction,
    get_all_predictions,
    get_prediction_by_id,
    delete_prediction,
    get_dashboard_stats
)

from utils.validation import validate_image_file
from utils.prediction import predictor_instance

# ---------------------------------------------------------------
# OPTIONAL GRAD-CAM IMPORT
# ---------------------------------------------------------------
# IMPORTANT:
# Your current predictor uses a TFLite model.
# Standard Keras Grad-CAM cannot directly use predictor_instance.model.
#
# We therefore keep Grad-CAM optional and disabled by default.
# Prediction will NEVER fail because Grad-CAM is unavailable.
# ---------------------------------------------------------------

try:
    from utils.gradcam import (
        generate_gradcam_heatmap,
        save_gradcam_overlay
    )

    GRADCAM_AVAILABLE = True

except Exception as e:

    print(
        f"[GRADCAM] Grad-CAM module unavailable: {e}",
        flush=True
    )

    GRADCAM_AVAILABLE = False


# ===============================================================
# FLASK APPLICATION
# ===============================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = SECRET_KEY
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

# Maximum upload size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ===============================================================
# DATABASE INITIALIZATION
# ===============================================================

with app.app_context():
    init_db()


# ===============================================================
# HELPERS
# ===============================================================

def load_disease_info():
    """
    Load disease knowledge-base JSON.
    """

    if not DISEASE_INFO_PATH.exists():
        return []

    try:

        with open(
            DISEASE_INFO_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception as e:

        print(
            f"[DISEASE INFO ERROR] {e}",
            flush=True
        )

        return []


def get_supported_class_names():
    """
    Load class names supported by the trained AI model.
    """

    if not CLASS_NAMES_PATH.exists():
        return []

    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            classes_dict = json.load(f)

        return list(classes_dict.values())

    except Exception as e:

        print(
            f"[CLASS NAMES ERROR] {e}",
            flush=True
        )

        return []


# ===============================================================
# HOME
# ===============================================================

@app.route("/")
def index():

    supported_classes = get_supported_class_names()

    model_loaded = predictor_instance.is_loaded

    return render_template(
        "index.html",
        model_loaded=model_loaded,
        supported_count=len(supported_classes)
    )


# ===============================================================
# DETECTION PAGE
# ===============================================================

@app.route("/detect")
def detect():

    model_loaded = predictor_instance.is_loaded

    return render_template(
        "detect.html",
        model_loaded=model_loaded
    )


# ===============================================================
# PREDICTION API
# ===============================================================

@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    def log(message):

        elapsed = time.time() - start_time

        print(
            f"[PREDICT +{elapsed:.2f}s] {message}",
            flush=True
        )

    log("=" * 60)
    log("PREDICTION REQUEST STARTED")
    log("=" * 60)

    try:

        # =======================================================
        # CHECK MODEL
        # =======================================================

        log("Checking model status...")

        if not predictor_instance.is_loaded:

            log(
                "Model is NOT loaded. "
                "Attempting to load model..."
            )

            predictor_instance.load_model_and_classes()

            log(
                "Model loading finished. "
                f"Loaded={predictor_instance.is_loaded}"
            )

        if not predictor_instance.is_loaded:

            log(
                "ERROR: Model could not be loaded."
            )

            return jsonify({

                "success": False,

                "error":
                    "AI model could not be loaded."

            }), 500

        log("Model is loaded and ready.")


        # =======================================================
        # CHECK IMAGE
        # =======================================================

        log("Checking uploaded image...")

        if "leaf_image" not in request.files:

            log(
                "ERROR: No leaf_image "
                "in request.files."
            )

            return jsonify({

                "success": False,

                "error":
                    "No image file uploaded."

            }), 400


        file = request.files["leaf_image"]


        log(
            f"Received file: {file.filename} | "
            f"content_type={file.content_type}"
        )


        if not file.filename:

            log("ERROR: Empty filename.")

            return jsonify({

                "success": False,

                "error":
                    "No image file selected."

            }), 400


        # =======================================================
        # VALIDATE IMAGE
        # =======================================================

        log("Starting image validation...")

        validation_start = time.time()

        is_valid, error_msg, pil_image = (
            validate_image_file(file)
        )

        validation_time = (
            time.time() - validation_start
        )

        log(
            "Image validation completed "
            f"in {validation_time:.2f}s | "
            f"valid={is_valid}"
        )


        if not is_valid:

            log(
                f"Image validation failed: "
                f"{error_msg}"
            )

            return jsonify({

                "success": False,

                "error": error_msg

            }), 400


        # =======================================================
        # SAVE ORIGINAL IMAGE
        # =======================================================

        log("Starting original image save...")

        unique_id = str(
            uuid.uuid4()
        )[:8]


        safe_filename = secure_filename(
            file.filename
        )


        # Fallback if secure_filename removes everything
        if not safe_filename:

            safe_filename = "leaf_image.jpg"


        filename = (
            f"{unique_id}_{safe_filename}"
        )


        leaf_upload_dir = (
            UPLOAD_FOLDER / "leaves"
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
            f"Original image saved: "
            f"{saved_image_path}"
        )


        # =======================================================
        # MODEL PREDICTION
        # =======================================================

        log(
            "=" * 60
        )

        log(
            "STARTING TFLITE MODEL PREDICTION"
        )

        log(
            "=" * 60
        )


        prediction_start = time.time()


        pred_result = (
            predictor_instance.predict(
                pil_image
            )
        )


        prediction_time = (
            time.time() - prediction_start
        )


        log(
            "MODEL PREDICTION FINISHED "
            f"in {prediction_time:.2f}s"
        )


        log(
            f"Prediction result: "
            f"{pred_result}"
        )


        # =======================================================
        # EXTRACT PREDICTION
        # =======================================================

        plant_name = pred_result.get(
            "plant_name",
            "Unknown Plant"
        )


        disease_name = pred_result.get(
            "disease_name",
            "Unknown Condition"
        )


        status = pred_result.get(
            "status",
            "Unknown"
        )


        confidence = pred_result.get(
            "confidence",
            0
        )


        raw_class = pred_result.get(
            "raw_class",
            "Unknown"
        )


        class_index = pred_result.get(
            "class_index",
            -1
        )


        log(
            f"Plant      : {plant_name}"
        )

        log(
            f"Disease    : {disease_name}"
        )

        log(
            f"Status     : {status}"
        )

        log(
            f"Confidence : {confidence}%"
        )

        log(
            f"Class Index: {class_index}"
        )


        # =======================================================
        # GRAD-CAM
        # =======================================================
        #
        # IMPORTANT:
        #
        # predictor_instance uses:
        #
        #     tf.lite.Interpreter
        #
        # It does NOT have:
        #
        #     predictor_instance.model
        #
        # Therefore the old code:
        #
        #     if predictor_instance.model is not None:
        #
        # was incorrect.
        #
        # We intentionally skip standard Grad-CAM here.
        #
        # The prediction itself remains fully functional.
        #
        # =======================================================

        gradcam_rel_path = None


        log(
            "Grad-CAM stage skipped."
        )

        log(
            "Reason: current predictor uses "
            "TFLite Interpreter, not a Keras model."
        )


        # =======================================================
        # DATABASE SAVE
        # =======================================================

        log(
            "=" * 60
        )

        log(
            "STARTING DATABASE SAVE"
        )

        log(
            "=" * 60
        )


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


        database_time = (
            time.time() - database_start
        )


        log(
            "Database save finished "
            f"in {database_time:.2f}s"
        )


        log(
            f"Prediction ID: {prediction_id}"
        )


        # =======================================================
        # TOTAL TIME
        # =======================================================

        total_time = (
            time.time() - start_time
        )


        log(
            "=" * 60
        )

        log(
            "PREDICTION COMPLETE"
        )

        log(
            f"MODEL TIME : {prediction_time:.2f}s"
        )

        log(
            f"DATABASE   : {database_time:.2f}s"
        )

        log(
            f"TOTAL TIME : {total_time:.2f}s"
        )

        log(
            "=" * 60
        )


        # =======================================================
        # RESPONSE
        # =======================================================

        return jsonify({

            "success": True,

            "prediction_id":
                prediction_id,

            "redirect_url":
                url_for(
                    "result",
                    id=prediction_id
                ),

            # Useful if frontend wants
            # immediate prediction data
            "prediction": {

                "plant_name":
                    plant_name,

                "disease_name":
                    disease_name,

                "status":
                    status,

                "confidence":
                    confidence,

                "raw_class":
                    raw_class,

                "class_index":
                    class_index

            }

        })


    # ===========================================================
    # RUNTIME ERROR
    # ===========================================================

    except RuntimeError as e:

        total_time = (
            time.time() - start_time
        )


        log(
            f"RUNTIME ERROR after "
            f"{total_time:.2f}s: {e}"
        )


        app.logger.exception(
            "Prediction RuntimeError"
        )


        return jsonify({

            "success": False,

            "error": str(e)

        }), 500


    # ===========================================================
    # GENERAL ERROR
    # ===========================================================

    except Exception as e:

        total_time = (
            time.time() - start_time
        )


        log(
            f"FATAL PREDICTION ERROR after "
            f"{total_time:.2f}s: {e}"
        )


        app.logger.exception(
            "Prediction pipeline failed"
        )


        return jsonify({

            "success": False,

            "error":
                f"Inference pipeline error: {str(e)}"

        }), 500


# ===============================================================
# RESULT PAGE
# ===============================================================

@app.route("/result/<int:id>")
def result(id):

    record = get_prediction_by_id(id)


    if not record:

        return render_template(
            "404.html",
            message="Prediction record not found."
        ), 404


    diseases = load_disease_info()


    # -----------------------------------------------------------
    # MATCH DISEASE INFORMATION
    # -----------------------------------------------------------

    matching_info = None


    plant_lower = (
        record["plant_name"]
        .lower()
    )


    disease_lower = (
        record["disease_name"]
        .lower()
    )


    for item in diseases:

        item_plant = (
            item["plant"]
            .lower()
        )

        item_disease = (
            item["disease"]
            .lower()
        )


        if (
            item_plant in plant_lower
            or plant_lower in item_plant
        ):

            if (
                item_disease in disease_lower
                or disease_lower in item_disease
            ):

                matching_info = item

                break


    # -----------------------------------------------------------
    # FALLBACK PLANT MATCH
    # -----------------------------------------------------------

    if not matching_info:

        for item in diseases:

            if (
                item["plant"]
                .lower()
                in plant_lower
            ):

                matching_info = item

                break


    # -----------------------------------------------------------
    # CONFIDENCE
    # -----------------------------------------------------------

    is_low_confidence = (
        record["confidence"]
        < (
            CONFIDENCE_THRESHOLD
            * 100.0
        )
    )


    return render_template(

        "result.html",

        record=record,

        disease_info=matching_info,

        is_low_confidence=is_low_confidence,

        confidence_threshold_pct=int(
            CONFIDENCE_THRESHOLD
            * 100.0
        )

    )


# ===============================================================
# DASHBOARD
# ===============================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html"
    )


# ===============================================================
# DASHBOARD API
# ===============================================================

@app.route("/api/dashboard-stats")
def api_dashboard_stats():

    stats = (
        get_dashboard_stats()
    )

    return jsonify(stats)


# ===============================================================
# HISTORY
# ===============================================================

@app.route("/history")
def history():

    return render_template(
        "history.html"
    )


# ===============================================================
# HISTORY API
# ===============================================================

@app.route(
    "/api/history",
    methods=["GET"]
)
def api_history():

    search = (
        request.args
        .get(
            "search",
            ""
        )
        .strip()
    )


    status_filter = (
        request.args
        .get(
            "status",
            "all"
        )
        .strip()
    )


    records = (
        get_all_predictions(
            search=search,
            status_filter=status_filter
        )
    )


    return jsonify({

        "success": True,

        "records": records

    })


# ===============================================================
# DELETE HISTORY RECORD
# ===============================================================

@app.route(
    "/api/history/<int:id>",
    methods=["DELETE"]
)
def api_delete_history(id):

    deleted = (
        delete_prediction(id)
    )


    if deleted:

        return jsonify({

            "success": True,

            "message":
                "Prediction record deleted successfully."

        })


    return jsonify({

        "success": False,

        "error":
            "Record not found or could not be deleted."

    }), 404


# ===============================================================
# ENCYCLOPEDIA
# ===============================================================

@app.route("/encyclopedia")
def encyclopedia():

    diseases = (
        load_disease_info()
    )


    supported_classes = (
        get_supported_class_names()
    )


    # -----------------------------------------------------------
    # UNIQUE PLANTS
    # -----------------------------------------------------------

    plants = sorted(
        list(
            set(
                item["plant"]
                for item in diseases
            )
        )
    )


    # -----------------------------------------------------------
    # MARK MODEL SUPPORT
    # -----------------------------------------------------------

    for item in diseases:

        item_plant = (
            item["plant"]
            .lower()
        )


        item_disease = (
            item["disease"]
            .lower()
        )


        item["is_supported"] = False


        for sc in supported_classes:

            sc_lower = (
                sc.lower()
                .replace(
                    "_",
                    " "
                )
            )


            if (
                item_plant in sc_lower
                and
                (
                    item_disease
                    in sc_lower
                    or
                    "healthy"
                    in item_disease
                )
            ):

                item["is_supported"] = True

                break


    return render_template(

        "encyclopedia.html",

        diseases=diseases,

        plants=plants

    )


# ===============================================================
# ENCYCLOPEDIA DETAIL
# ===============================================================

@app.route(
    "/encyclopedia/<disease_id>"
)
def encyclopedia_detail(
    disease_id
):

    diseases = (
        load_disease_info()
    )


    disease_entry = next(

        (
            item
            for item in diseases
            if item["id"] == disease_id
        ),

        None

    )


    if not disease_entry:

        return render_template(

            "404.html",

            message=
                "Disease entry not found in encyclopedia."

        ), 404


    supported_classes = (
        get_supported_class_names()
    )


    is_supported = False


    for sc in supported_classes:

        sc_lower = (
            sc.lower()
            .replace(
                "_",
                " "
            )
        )


        if (

            disease_entry["plant"]
            .lower()
            in sc_lower

            and

            disease_entry["disease"]
            .lower()
            in sc_lower

        ):

            is_supported = True

            break


    return render_template(

        "disease.html",

        disease=disease_entry,

        is_supported=is_supported

    )


# ===============================================================
# ABOUT
# ===============================================================

@app.route("/about")
def about():

    model_loaded = (
        predictor_instance.is_loaded
    )


    metadata = {}


    metadata_path = (
        MODEL_PATH.parent
        / "model_metadata.json"
    )


    if metadata_path.exists():

        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as f:

                metadata = json.load(f)

        except Exception as e:

            print(
                f"[ABOUT] Metadata error: {e}",
                flush=True
            )


    return render_template(

        "about.html",

        model_loaded=model_loaded,

        metadata=metadata

    )


# ===============================================================
# HEALTH CHECK
# ===============================================================

@app.route("/health")
def health():

    db_ok = (
        DATABASE_PATH.exists()
    )


    model_ok = (
        predictor_instance.is_loaded
    )


    status_str = (
        "ok"
        if (
            db_ok
            and
            model_ok
        )
        else
        "degraded"
    )


    return jsonify({

        "status":
            status_str,

        "model_loaded":
            model_ok,

        "database":
            "ok"
            if db_ok
            else
            "missing",

        "confidence_threshold":
            CONFIDENCE_THRESHOLD

    })


# ===============================================================
# SERVE UPLOADED MEDIA
# ===============================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(filename):

    return send_from_directory(

        app.config["UPLOAD_FOLDER"],

        filename

    )


# ===============================================================
# PWA MANIFEST
# ===============================================================

@app.route("/manifest.json")
def manifest():

    return send_from_directory(
        ".",
        "manifest.json"
    )


# ===============================================================
# SERVICE WORKER
# ===============================================================

@app.route("/service-worker.js")
def service_worker():

    return send_from_directory(
        ".",
        "service-worker.js"
    )


# ===============================================================
# ERROR HANDLERS
# ===============================================================

@app.errorhandler(404)
def page_not_found(e):

    return render_template(

        "404.html",

        message=
            "The requested page could not be found."

    ), 404


@app.errorhandler(500)
def server_error(e):

    return render_template(

        "404.html",

        message=
            "An unexpected server error occurred."

    ), 500


# ===============================================================
# APPLICATION START
# ===============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "PLANT DISEASE DETECTION "
        "& CLASSIFICATION FLASK APP"
    )

    print(
        "Model Status: "
        + (
            "LOADED & READY"
            if predictor_instance.is_loaded
            else
            "NOT LOADED"
        )
    )

    print(
        "Grad-CAM Available: "
        + str(GRADCAM_AVAILABLE)
    )

    print(
        "Running on "
        "http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(

        host="0.0.0.0",

        port=5000,

        debug=DEBUG

    )