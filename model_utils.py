"""
Model utilities for image colorization using U-Net architecture.
Includes model loading, preprocessing, and inference functions.
"""

import tensorflow as tf
import cv2
import numpy as np
import os

# Handle Keras 3.x compatibility (TensorFlow 2.20+ uses Keras 3)
try:
    from keras import layers, Model
    from keras.applications import VGG16
except ImportError:
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import VGG16

IMG_SIZE = 256


def prep_L_to_RGB(L):
    """Convert single-channel L (grayscale) to 3-channel RGB for VGG16 input."""
    return tf.image.grayscale_to_rgb(L)


def build_unet(img_size=IMG_SIZE):
    """
    Build standard U-Net model for image colorization.
    
    Args:
        img_size: Input image size (default 256)
    
    Returns:
        Compiled Keras Model
    """
    inp = layers.Input((img_size, img_size, 1))
    c1 = layers.Conv2D(64, 3, padding='same', activation='relu')(inp)
    p1 = layers.MaxPool2D()(c1)
    c2 = layers.Conv2D(128, 3, padding='same', activation='relu')(p1)
    p2 = layers.MaxPool2D()(c2)
    b = layers.Conv2D(256, 3, padding='same', activation='relu')(p2)
    u1 = layers.UpSampling2D()(b)
    m1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv2D(128, 3, padding='same', activation='relu')(m1)
    u2 = layers.UpSampling2D()(c3)
    m2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv2D(64, 3, padding='same', activation='relu')(m2)
    out = layers.Conv2D(2, 1, activation='tanh')(c4)
    return Model(inp, out)


def build_vgg_unet(img_size=IMG_SIZE):
    """
    Build VGG16-U-Net model with transfer learning for better colorization.
    
    Args:
        img_size: Input image size (default 256)
    
    Returns:
        Compiled Keras Model
    """
    # Input is single-channel L; convert to 3-channel for VGG
    inp_L = layers.Input((img_size, img_size, 1))
    x = layers.Lambda(prep_L_to_RGB)(inp_L)
    
    # Pretrained VGG16 as encoder
    vgg = VGG16(weights='imagenet', include_top=False, input_tensor=x)
    for layer in vgg.layers:
        layer.trainable = False  # freeze encoder

    # Skip connections from VGG blocks
    skips = [
        vgg.get_layer(name).output for name in
        ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool']
    ]
    bottleneck = vgg.get_layer('block5_pool').output  # shape (8,8,512)

    # Decoder: up + concat + conv
    u = bottleneck
    filters = [512, 256, 128, 64]
    for i, skip in enumerate(reversed(skips)):
        u = layers.UpSampling2D()(u)
        u = layers.Concatenate()([u, skip])
        u = layers.Conv2D(filters[i], 3, padding='same', activation='relu')(u)
        u = layers.Conv2D(filters[i], 3, padding='same', activation='relu')(u)

    # Final up to original resolution
    u = layers.UpSampling2D()(u)
    u = layers.Conv2D(64, 3, padding='same', activation='relu')(u)
    out_ab = layers.Conv2D(2, 1, activation='tanh')(u)

    return Model(inp_L, out_ab)


def load_model(model_path=None, model_type='vgg_unet'):
    """
    Load pre-trained model from file or build new model.
    
    Args:
        model_path: Path to saved model file (.h5)
        model_type: 'vgg_unet' or 'unet' (default: 'vgg_unet')
    
    Returns:
        Loaded or newly built Keras Model
    """
    if model_type == 'vgg_unet':
        model = build_vgg_unet()
    else:
        model = build_unet()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=lambda y_true, y_pred: tf.reduce_mean(tf.square(y_true - y_pred))
    )
    
    # Load weights if model path provided
    if model_path and os.path.exists(model_path):
        try:
            model.load_weights(model_path)
            print(f"Loaded model weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights from {model_path}: {e}")
            print("Using untrained model")
    else:
        print(f"Model path not found: {model_path}")
        print("Using untrained model (will need training)")
    
    return model


def preprocess_image(image_path_or_array, is_file_path=True):
    """
    Preprocess image for model input.
    Converts image to grayscale, resizes, and normalizes.
    
    Args:
        image_path_or_array: Path to image file or numpy array
        is_file_path: If True, treats first arg as file path; else as array
    
    Returns:
        Preprocessed image array ready for model input (1, 256, 256, 1)
    """
    if is_file_path:
        # Read image
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"Could not read image from {image_path_or_array}")
    else:
        img = image_path_or_array.copy()
    
    # Convert BGR to RGB if color image
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to grayscale if color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # Resize to model input size
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0, 1]
    gray = gray.astype(np.float32) / 255.0
    
    # Add channel dimension and batch dimension
    gray = gray[..., np.newaxis]  # (256, 256, 1)
    gray = np.expand_dims(gray, axis=0)  # (1, 256, 256, 1)
    
    return gray


def lab_to_rgb_corrected(L_gray, pred_ab):
    """
    Corrected LAB to RGB conversion.
    
    The notebook version had incorrect scaling. This version properly:
    - Converts normalized L channel [0, 1] to LAB L range [0, 100]
    - Denormalizes a, b channels from [-1, 1] to LAB range [0, 255] then to [-128, 127]
    - Uses OpenCV for accurate LAB to RGB conversion
    
    Args:
        L_gray: Grayscale L channel, shape (H, W, 1) or (1, H, W, 1), normalized [0, 1]
        pred_ab: Predicted ab channels, shape (H, W, 2) or (1, H, W, 2), normalized [-1, 1]
    
    Returns:
        RGB image array, shape (H, W, 3), values in [0, 255]
    """
    # Remove batch dimension if present
    if len(L_gray.shape) == 4:
        L_gray = L_gray[0]  # (H, W, 1)
    if len(pred_ab.shape) == 4:
        pred_ab = pred_ab[0]  # (H, W, 2)
    
    # Convert L from [0, 1] to [0, 100] (LAB L channel range)
    # Squeeze if single channel dimension exists
    if L_gray.shape[-1] == 1:
        L_gray = L_gray.squeeze(-1)
    L = L_gray * 100.0
    
    # Denormalize a, b from [-1, 1] to LAB range
    # LAB a and b range is [-127, 127] in OpenCV, stored as [0, 255]
    # So: (tanh_output * 127) + 128 gives [1, 255], which maps to [-126, 127]
    # Better: tanh_output * 127 + 128, then clip to [0, 255]
    a = (pred_ab[:, :, 0] * 127.0 + 128.0).clip(0, 255)
    b = (pred_ab[:, :, 1] * 127.0 + 128.0).clip(0, 255)
    
    # Stack L, a, b channels - ensure L is 2D
    if len(L.shape) == 2:
        L = L[..., np.newaxis]
    if len(a.shape) == 2:
        a = a[..., np.newaxis]
    if len(b.shape) == 2:
        b = b[..., np.newaxis]
    
    lab = np.concatenate([L, a, b], axis=-1).astype(np.uint8)
    
    # Convert LAB to RGB using OpenCV (expects uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return rgb


def predict_colorization(model, preprocessed_image):
    """
    Run colorization inference on preprocessed image (optimized for speed).
    
    Args:
        model: Trained Keras model
        preprocessed_image: Preprocessed image array (1, 256, 256, 1)
    
    Returns:
        Colorized RGB image array (256, 256, 3), values in [0, 255]
    """
    # Use __call__ instead of predict for faster inference (no overhead)
    pred_ab = model(preprocessed_image, training=False)
    
    # Convert tensor to numpy if needed
    if hasattr(pred_ab, 'numpy'):
        pred_ab = pred_ab.numpy()
    
    # Extract L channel from input (it's normalized [0, 1])
    L_gray = preprocessed_image[0]  # Remove batch dim: (256, 256, 1)
    
    # Convert LAB to RGB
    colorized = lab_to_rgb_corrected(L_gray, pred_ab)
    
    return colorized

