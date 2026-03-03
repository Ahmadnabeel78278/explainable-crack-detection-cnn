import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNet

# Helper functions for Lambda layers (import tf inside to avoid serialisation issues)
def _mean_keepdims(x):
    import tensorflow as tf
    return tf.reduce_mean(x, axis=-1, keepdims=True)

def _max_keepdims(x):
    import tensorflow as tf
    return tf.reduce_max(x, axis=-1, keepdims=True)

def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    shared_dense_one = layers.Dense(channel//ratio, activation='relu')
    shared_dense_two = layers.Dense(channel)

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_dense_one(avg_pool)
    avg_pool = shared_dense_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1,1,channel))(max_pool)
    max_pool = shared_dense_one(max_pool)
    max_pool = shared_dense_two(max_pool)

    attention = layers.Add()([avg_pool, max_pool])
    attention = layers.Activation('sigmoid')(attention)

    return layers.Multiply()([input_feature, attention])

def spatial_attention(input_feature):
    # Use helper functions that import tensorflow internally
    avg_pool = layers.Lambda(_mean_keepdims)(input_feature)
    max_pool = layers.Lambda(_max_keepdims)(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([input_feature, attention])

def cbam_block(input_feature):
    x = channel_attention(input_feature)
    x = spatial_attention(x)
    return x

def build_model(input_shape=(224,224,3)):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base layers initially

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = cbam_block(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model