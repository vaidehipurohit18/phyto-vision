import tensorflow as tf
from training.attention import CBAMLayer

print("Loading fine-tuned model...")

model = tf.keras.models.load_model(
    "model/plant_disease_finetuned.keras",
    custom_objects={"CBAMLayer": CBAMLayer},
    compile=False
)

print("Converting model to TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model/plant_disease.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite conversion complete!")