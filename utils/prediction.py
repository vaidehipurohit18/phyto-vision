import os

# ==============================================================
# RENDER / CPU SETTINGS
# ==============================================================

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import tensorflow as tf


# ==============================================================
# PROJECT ROOT
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from config import (
    MODEL_DIR,
    CLASS_NAMES_PATH,
    CONFIDENCE_THRESHOLD,
)

from utils.preprocessing import preprocess_leaf_image


class PlantDiseasePredictor:

    _instance = None

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super(
                PlantDiseasePredictor,
                cls
            ).__new__(cls)

            cls._instance.interpreter = None
            cls._instance.input_details = None
            cls._instance.output_details = None
            cls._instance.class_names = {}
            cls._instance.is_loaded = False

            cls._instance.load_model_and_classes()

        return cls._instance

    # ==========================================================
    # LOAD TFLITE MODEL
    # ==========================================================

    def load_model_and_classes(self):

        tflite_path = MODEL_DIR / "plant_disease.tflite"

        if not tflite_path.exists():

            print(
                f"[PREDICTOR ERROR] TFLite model not found: "
                f"{tflite_path}",
                flush=True
            )

            self.is_loaded = False
            return

        if not CLASS_NAMES_PATH.exists():

            print(
                f"[PREDICTOR ERROR] Class names not found: "
                f"{CLASS_NAMES_PATH}",
                flush=True
            )

            self.is_loaded = False
            return

        try:

            print(
                "[PREDICTOR] Loading TFLite model...",
                flush=True
            )

            print(
                f"[PREDICTOR] Model path: {tflite_path}",
                flush=True
            )

            self.interpreter = tf.lite.Interpreter(
                model_path=str(tflite_path),
                num_threads=1
            )

            print(
                "[PREDICTOR] Allocating TFLite tensors...",
                flush=True
            )

            self.interpreter.allocate_tensors()

            self.input_details = (
                self.interpreter.get_input_details()
            )

            self.output_details = (
                self.interpreter.get_output_details()
            )

            print(
                "[PREDICTOR] Input shape:",
                self.input_details[0]["shape"],
                flush=True
            )

            print(
                "[PREDICTOR] Output shape:",
                self.output_details[0]["shape"],
                flush=True
            )

            with open(CLASS_NAMES_PATH, "r") as f:

                raw_classes = json.load(f)

            self.class_names = {
                int(k): v
                for k, v in raw_classes.items()
            }

            self.is_loaded = True

            print(
                "[PREDICTOR] Successfully loaded TFLite model "
                f"and {len(self.class_names)} classes.",
                flush=True
            )

        except Exception as e:

            print(
                "[PREDICTOR ERROR] Failed loading TFLite model:",
                str(e),
                flush=True
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

            print(
                "[PREDICTOR] Model not loaded. "
                "Attempting reload...",
                flush=True
            )

            self.load_model_and_classes()

            if not self.is_loaded:

                raise RuntimeError(
                    "AI model is not loaded."
                )

        if pil_image is None:

            raise ValueError(
                "No image was provided."
            )

        # ------------------------------------------------------
        # PREPROCESS
        # ------------------------------------------------------

        print(
            "[PREDICTOR] Starting image preprocessing...",
            flush=True
        )

        input_batch = preprocess_leaf_image(
            pil_image
        )

        input_batch = np.asarray(
            input_batch,
            dtype=np.float32
        )

        print(
            "[PREDICTOR] Input shape:",
            input_batch.shape,
            flush=True
        )

        print(
            "[PREDICTOR] Input dtype:",
            input_batch.dtype,
            flush=True
        )

        # ------------------------------------------------------
        # VERIFY INPUT SHAPE
        # ------------------------------------------------------

        if input_batch.shape != (
            1,
            224,
            224,
            3
        ):

            print(
                "[PREDICTOR] Reshaping input...",
                flush=True
            )

            input_batch = input_batch.reshape(
                1,
                224,
                224,
                3
            )

        # ------------------------------------------------------
        # TENSOR INDICES
        # ------------------------------------------------------

        input_index = (
            self.input_details[0]["index"]
        )

        output_index = (
            self.output_details[0]["index"]
        )

        # ------------------------------------------------------
        # TFLITE INFERENCE DIAGNOSTICS
        # ------------------------------------------------------

        print(
            "[PREDICTOR] Starting TFLite inference...",
            flush=True
        )

        print(
            "[PREDICTOR] BEFORE set_tensor",
            flush=True
        )

        self.interpreter.set_tensor(
            input_index,
            input_batch
        )

        print(
            "[PREDICTOR] AFTER set_tensor",
            flush=True
        )

        print(
            "[PREDICTOR] BEFORE invoke",
            flush=True
        )

        self.interpreter.invoke()

        print(
            "[PREDICTOR] AFTER invoke",
            flush=True
        )

        predictions = (
            self.interpreter.get_tensor(
                output_index
            )[0]
        )

        print(
            "[PREDICTOR] AFTER get_tensor",
            flush=True
        )

        print(
            "[PREDICTOR] TFLite inference completed.",
            flush=True
        )

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

        (
            plant_name,
            disease_name,
            status
        ) = self.parse_class_name(
            raw_class
        )

        # ------------------------------------------------------
        # LOW CONFIDENCE
        # ------------------------------------------------------

        is_low_confidence = (
            confidence_val
            < CONFIDENCE_THRESHOLD
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

            (
                p_name,
                d_name,
                c_status
            ) = self.parse_class_name(
                c_raw
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
        # DEBUG OUTPUT
        # ------------------------------------------------------

        print(
            "\n" + "=" * 70,
            flush=True
        )

        print(
            "[PREDICTION]",
            flush=True
        )

        print(
            f"Plant      : {plant_name}",
            flush=True
        )

        print(
            f"Disease    : {disease_name}",
            flush=True
        )

        print(
            f"Status     : {status}",
            flush=True
        )

        print(
            f"Confidence : {confidence_pct}%",
            flush=True
        )

        print(
            "=" * 70,
            flush=True
        )

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

            "top_3": top_5_predictions[:3]

        }


# ==============================================================
# GLOBAL INSTANCE
# ==============================================================

predictor_instance = PlantDiseasePredictor()