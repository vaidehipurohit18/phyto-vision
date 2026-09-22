import tensorflow as tf

# Import custom layer BEFORE loading model
from training.attention import CBAMLayer

MODEL_PATH = "model/plant_disease_finetuned.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "CBAMLayer": CBAMLayer
    },
    compile=False
)

print("\n" + "=" * 80)
print("MODEL SUMMARY")
print("=" * 80)

model.summary(expand_nested=True)

print("\n" + "=" * 80)
print("TOP LEVEL LAYERS")
print("=" * 80)

for i, layer in enumerate(model.layers):
    print(f"{i}: {layer.name} | {type(layer).__name__}")

print("\n" + "=" * 80)
print("NESTED LAYERS")
print("=" * 80)

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        print(f"\nNested model: {layer.name}")
        for i, sublayer in enumerate(layer.layers):
            print(
                f"  {i}: {sublayer.name} | "
                f"{type(sublayer).__name__} | "
                f"Output: {sublayer.output.shape}"
            )