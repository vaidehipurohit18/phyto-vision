import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from training.attention import CBAMLayer

def build_attention_model(input_shape=(224, 224, 3), num_classes=38, trainable_base_layers=0):
    """
    Builds an Attention-Based CNN model using MobileNetV2 feature extractor and CBAM module.
    
    Architecture:
    Input (224x224x3) -> MobileNetV2 -> CBAM Attention -> GAP -> Dropout -> Dense (Softmax)
    """
    # Base MobileNetV2 model
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    if trainable_base_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-trainable_base_layers]:
            layer.trainable = False

    inputs = layers.Input(shape=input_shape)
    
    # Preprocessing matching MobileNetV2 expectations [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Feature extraction via MobileNetV2
    features = base_model(x, training=False)
    
    # CBAM Attention Module integration
    attention_features = CBAMLayer(ratio=8, kernel_size=7, name='cbam_attention')(features)
    
    # Classification Head
    gap = layers.GlobalAveragePooling2D(name='global_avg_pool')(attention_features)
    dropout = layers.Dropout(0.3, name='dropout')(gap)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(dropout)

    model = models.Model(inputs=inputs, outputs=outputs, name='MobileNetV2_CBAM_Plant_Disease')
    return model, base_model
