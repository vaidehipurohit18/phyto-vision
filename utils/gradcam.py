import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import sys

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import IMAGE_SIZE, UPLOAD_FOLDER


def generate_gradcam_heatmap(model, input_batch, target_class_idx):
    """
    Generate Grad-CAM heatmap using the last convolutional layer
    (Conv_1) inside the MobileNetV2 backbone.
    """

    try:
        # Get MobileNetV2 backbone
        base_model = model.get_layer("mobilenetv2_1.00_224")

        # Confirmed last convolutional layer
        last_conv_layer = base_model.get_layer("Conv_1")

        # Model that gives Conv_1 feature maps
        conv_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=last_conv_layer.output
        )

        with tf.GradientTape() as tape:

            # Get feature maps from Conv_1
            conv_outputs = conv_model(input_batch)

            # Pass feature maps through the remaining pipeline
            x = conv_outputs

            # CBAM attention
            x = model.get_layer("cbam_attention")(x)

            # Global average pooling
            x = model.get_layer("global_avg_pool")(x)

            # Dropout - inference mode
            x = model.get_layer("dropout")(x, training=False)

            # Final prediction
            predictions = model.get_layer("predictions")(x)

            # Score for predicted/target class
            loss = predictions[:, target_class_idx]

        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            print("[GRAD-CAM ERROR] Gradients are None.")
            return None

        # Average gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        # Remove batch dimension
        conv_outputs = conv_outputs[0]

        # Weight feature maps using gradient importance
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

        heatmap = tf.squeeze(heatmap)

        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)

        # Normalize safely
        max_value = tf.reduce_max(heatmap)

        if max_value > 0:
            heatmap = heatmap / max_value

        return heatmap.numpy()

    except Exception as e:
        print(f"[GRAD-CAM WARNING] Could not compute Grad-CAM heatmap: {e}")
        return None

def save_gradcam_overlay(pil_image, heatmap, output_filename):
    """
    Overlays Grad-CAM heatmap onto original PIL leaf image and saves to disk.
    
    Returns relative output path if successful, else None.
    """
    if heatmap is None:
        return None

    try:
        # Resize original image to target size
        orig_img = np.array(pil_image.resize(IMAGE_SIZE), dtype=np.uint8)
        orig_bgr = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)

        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, IMAGE_SIZE)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply JET colormap
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend heatmap with original image (40% heatmap, 60% original)
        superimposed = cv2.addWeighted(orig_bgr, 0.6, heatmap_color, 0.4, 0)

        # Save overlaid image
        out_dir = UPLOAD_FOLDER / 'gradcam'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = out_dir / output_filename
        cv2.imwrite(str(save_path), superimposed)

        return f"uploads/gradcam/{output_filename}"
    except Exception as e:
        print(f"[GRAD-CAM OVERLAY ERROR] Failed overlay generation: {e}")
        return None
if __name__ == "__main__":
    print("=" * 60)
    print("FINDING GRAD-CAM TARGET LAYER")
    print("=" * 60)

    from training.attention import CBAMLayer

    MODEL_PATH = "model/plant_disease_finetuned.keras"

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "CBAMLayer": CBAMLayer
        },
        compile=False
    )

    print("\nModel loaded successfully.\n")

    # Get MobileNetV2 backbone
    base_model = model.get_layer("mobilenetv2_1.00_224")

    print("Searching for Conv2D layers inside MobileNetV2...\n")

    conv_layers = []

    for i, layer in enumerate(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layers.append((i, layer.name, layer.output.shape))

    print("Last 10 Conv2D layers:\n")

    for item in conv_layers[-10:]:
        print(item)

    print("\nLast Conv2D layer:")
    print(conv_layers[-1])

if __name__ == "__main__":

    print("=" * 60)
    print("GRAD-CAM REAL TEST")
    print("=" * 60)

    from training.attention import CBAMLayer

    MODEL_PATH = "model/plant_disease_finetuned.keras"

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "CBAMLayer": CBAMLayer
        },
        compile=False
    )

    print("Model loaded successfully.")

    # Change this path to an actual leaf image
    IMAGE_PATH = "test_image.jpg"

    if not os.path.exists(IMAGE_PATH):
        print(f"\nERROR: Image not found: {IMAGE_PATH}")
        print("Put one leaf image in the project root and name it test_image.jpg")

    else:
        print(f"\nLoading image: {IMAGE_PATH}")

        pil_image = Image.open(IMAGE_PATH).convert("RGB")

        # Resize and prepare image
        image = pil_image.resize(IMAGE_SIZE)

        image_array = np.array(image, dtype=np.float32)

        # MobileNetV2 preprocessing
        image_array = image_array / 127.5 - 1

        # Add batch dimension
        input_batch = np.expand_dims(image_array, axis=0)

        # Prediction
        predictions = model.predict(input_batch, verbose=0)

        predicted_class_idx = np.argmax(predictions[0])

        confidence = predictions[0][predicted_class_idx] * 100

        print(f"\nPredicted class index: {predicted_class_idx}")
        print(f"Confidence: {confidence:.2f}%")

        print("\nGenerating Grad-CAM heatmap...")

        heatmap = generate_gradcam_heatmap(
            model,
            input_batch,
            predicted_class_idx
        )

        if heatmap is None:
            print("\nGRAD-CAM FAILED")

        else:
            print("Grad-CAM heatmap generated successfully.")
            print(f"Heatmap shape: {heatmap.shape}")
            print(f"Heatmap min: {heatmap.min():.4f}")
            print(f"Heatmap max: {heatmap.max():.4f}")

            output_path = save_gradcam_overlay(
                pil_image,
                heatmap,
                "gradcam_test.jpg"
            )

            if output_path:
                print("\nSUCCESS!")
                print(f"Saved Grad-CAM overlay to: {output_path}")
            else:
                print("\nOverlay generation failed.")