import os

# Keep TensorFlow CPU resource usage low on Render Free.
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import tensorflow as tf

# Limit TensorFlow CPU threads.
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass
import json
from pathlib import Path
import sys

try:
    import numpy as np
except Exception:
    np = None

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False


# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import MODEL_PATH, CLASS_NAMES_PATH, CONFIDENCE_THRESHOLD
from utils.preprocessing import preprocess_leaf_image


class PlantDiseasePredictor:

    _instance = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super(
                PlantDiseasePredictor,
                cls
            ).__new__(cls)

            cls._instance.model = None
            cls._instance.class_names = {}
            cls._instance.is_loaded = False

            cls._instance.load_model_and_classes()

        return cls._instance

    # ==========================================================
    # LOAD MODEL
    # ==========================================================

    def load_model_and_classes(self):

        if (
            not HAS_TENSORFLOW
            or not MODEL_PATH.exists()
            or not CLASS_NAMES_PATH.exists()
        ):

            self.is_loaded = False
            return

        try:

            from training.attention import CBAMLayer

            custom_objects = {
                "CBAMLayer": CBAMLayer
            }

            self.model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects=custom_objects
            )

            with open(CLASS_NAMES_PATH, "r") as f:

                raw_classes = json.load(f)

            self.class_names = {
                int(k): v
                for k, v in raw_classes.items()
            }

            self.is_loaded = True

            print(
                f"[PREDICTOR] Successfully loaded model "
                f"and {len(self.class_names)} class categories."
            )

            print("[PREDICTOR] CLASS MAPPING:")

            for idx, name in self.class_names.items():

                print(
                    f"    {idx}: {name}"
                )

        except Exception as e:

            print(
                f"[PREDICTOR ERROR] Failed loading model: {e}"
            )

            self.is_loaded = False

    # ==========================================================
    # PARSE CLASS NAME
    # ==========================================================

    def parse_class_name(self, raw_class_str):

        if (
            not raw_class_str
            or raw_class_str == "Unknown"
        ):

            return (
                "Unknown Plant",
                "Unknown Condition",
                "Unknown"
            )

        if "___" in raw_class_str:

            plant_raw, condition_raw = (
                raw_class_str.split(
                    "___",
                    1
                )
            )

        else:

            plant_raw = raw_class_str
            condition_raw = "healthy"

        plant_name = (
            plant_raw
            .replace("_", " ")
            .replace(",", ", ")
            .strip()
            .title()
        )

        disease_name = (
            condition_raw
            .replace("_", " ")
            .strip()
            .title()
        )

        if disease_name.lower() == "healthy":

            disease_name = "Healthy"
            status = "Healthy"

        else:

            status = "Diseased"

        return (
            plant_name,
            disease_name,
            status
        )

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(self, pil_image):

        if not self.is_loaded:

            self.load_model_and_classes()

            if not self.is_loaded:

                raise RuntimeError(
                    "AI model is not trained yet."
                )

        if np is None:

            raise RuntimeError(
                "NumPy is not installed."
            )

        # ------------------------------------------------------
        # PREPROCESS
        # ------------------------------------------------------

        input_batch = preprocess_leaf_image(
            pil_image
        )

        # ------------------------------------------------------
        # MODEL PREDICTION
        # ------------------------------------------------------

        predictions = self.model.predict(
            input_batch,
            verbose=0
        )[0]

        # ------------------------------------------------------
        # TOP CLASS
        # ------------------------------------------------------

        top_idx = int(
            np.argmax(predictions)
        )

        confidence_val = float(
            predictions[top_idx]
        )

        confidence_pct = round(
            confidence_val * 100.0,
            2
        )

        raw_class = self.class_names.get(
            top_idx,
            "Unknown"
        )

        plant_name, disease_name, status = (
            self.parse_class_name(
                raw_class
            )
        )

        # ------------------------------------------------------
        # LOW CONFIDENCE
        # ------------------------------------------------------

        is_low_confidence = (
            confidence_val < CONFIDENCE_THRESHOLD
        )

        # ------------------------------------------------------
        # TOP 5
        # ------------------------------------------------------

        top_5_indices = np.argsort(
            predictions
        )[-5:][::-1]

        top_5_predictions = []

        for idx in top_5_indices:

            idx = int(idx)

            c_raw = self.class_names.get(
                idx,
                "Unknown"
            )

            p_name, d_name, c_status = (
                self.parse_class_name(
                    c_raw
                )
            )

            score = float(
                predictions[idx]
            )

            top_5_predictions.append({

                "class_index": idx,

                "raw_class": c_raw,

                "plant_name": p_name,

                "disease_name": d_name,

                "status": c_status,

                "confidence": round(
                    score * 100.0,
                    2
                ),

                "confidence_raw": score

            })

        # ------------------------------------------------------
        # PRINT DEBUG INFORMATION
        # ------------------------------------------------------

        print("\n" + "=" * 70)

        print("[PREDICTION]")

        print(
            f"Top class index : {top_idx}"
        )

        print(
            f"Top class       : {raw_class}"
        )

        print(
            f"Plant           : {plant_name}"
        )

        print(
            f"Disease         : {disease_name}"
        )

        print(
            f"Status          : {status}"
        )

        print(
            f"Confidence      : {confidence_pct}%"
        )

        print(
            f"Low confidence  : {is_low_confidence}"
        )

        print(
            f"Threshold       : "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}%"
        )

        print("\nTOP 5 PREDICTIONS:")

        for item in top_5_predictions:

            print(
                f"   "
                f"{item['confidence']:6.2f}%  "
                f"{item['raw_class']}"
            )

        print("=" * 70)

        # ------------------------------------------------------
        # RETURN
        # ------------------------------------------------------

        return {

            "raw_class": raw_class,

            "class_index": top_idx,

            "plant_name": plant_name,

            "disease_name": disease_name,

            "status": status,

            "confidence": confidence_pct,

            "confidence_raw": confidence_val,

            "is_low_confidence": is_low_confidence,

            "confidence_threshold": (
                CONFIDENCE_THRESHOLD * 100.0
            ),

            "top_5": top_5_predictions,

            # Keep top_3 for compatibility
            "top_3": top_5_predictions[:3]

        }


# ==============================================================
# GLOBAL INSTANCE
# ==============================================================

predictor_instance = PlantDiseasePredictor()