import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="CustomLayers")
class CBAMLayer(layers.Layer):
    """
    Convolutional Block Attention Module (CBAM)
    Consists of Channel Attention Module followed by Spatial Attention Module.
    Reference: Woo et al., 'CBAM: Convolutional Block Attention Module', ECCV 2018.
    """
    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(CBAMLayer, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        channel = input_shape[-1]
        
        # Shared MLP for Channel Attention
        self.shared_mlp_1 = layers.Dense(
            channel // self.ratio, 
            activation='relu', 
            kernel_initializer='he_normal', 
            use_bias=True, 
            bias_initializer='zeros'
        )
        self.shared_mlp_2 = layers.Dense(
            channel, 
            kernel_initializer='he_normal', 
            use_bias=True, 
            bias_initializer='zeros'
        )

        # 2D Convolution for Spatial Attention
        self.conv2d = layers.Conv2D(
            filters=1, 
            kernel_size=self.kernel_size, 
            strides=1, 
            padding='same', 
            activation='sigmoid', 
            kernel_initializer='he_normal', 
            use_bias=False
        )
        
        super(CBAMLayer, self).build(input_shape)

    def channel_attention(self, input_feature):
        # Global Average Pooling & Global Max Pooling
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keepdims=True)

        # Shared MLP forward pass
        avg_out = self.shared_mlp_2(self.shared_mlp_1(avg_pool))
        max_out = self.shared_mlp_2(self.shared_mlp_1(max_pool))

        # Add feature representations & apply Sigmoid
        channel_att = tf.nn.sigmoid(avg_out + max_out)
        return input_feature * channel_att

    def spatial_attention(self, input_feature):
        # Channel-wise Avg Pooling & Max Pooling
        avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)

        # Concatenate along channel axis
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        # Convolution & Sigmoid activation
        spatial_att = self.conv2d(concat)
        return input_feature * spatial_att

    def call(self, inputs):
        # Forward pass through Channel Attention then Spatial Attention
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super(CBAMLayer, self).get_config()
        config.update({
            'ratio': self.ratio,
            'kernel_size': self.kernel_size
        })
        return config
