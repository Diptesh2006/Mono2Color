"""
Flask backend for AI-based image colorization web application.
Provides REST API endpoint for colorizing black-and-white images.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
from PIL import Image
import io
from model_utils import load_model, preprocess_image, predict_colorization

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Load model (try VGG-UNet first, fallback to U-Net)
MODEL = None
MODEL_TYPE = None

def load_colorization_model():
    """Load the colorization model with optimizations."""
    global MODEL, MODEL_TYPE
    
    import tensorflow as tf
    
    # Enable optimizations for faster inference
    try:
        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
    except:
        pass  # XLA might not be available on all systems
    
    # Try to load VGG-UNet first (better results)
    vgg_model_path = os.path.join(MODEL_FOLDER, 'vgg_unet_best.h5')
    if os.path.exists(vgg_model_path):
        try:
            MODEL = load_model(vgg_model_path, model_type='vgg_unet')
            MODEL_TYPE = 'vgg_unet'
            print("[OK] Loaded VGG-UNet model")
            print(f"  Model path: {vgg_model_path}")
        except Exception as e:
            print(f"Error loading VGG-UNet: {e}")
            MODEL = None
    else:
        print(f"VGG-UNet model not found at: {vgg_model_path}")
    
    # Fallback to standard U-Net if VGG-UNet failed
    if MODEL is None:
        unet_model_path = os.path.join(MODEL_FOLDER, 'unet_best.h5')
        if os.path.exists(unet_model_path):
            try:
                MODEL = load_model(unet_model_path, model_type='unet')
                MODEL_TYPE = 'unet'
                print("[OK] Loaded U-Net model")
            except Exception as e:
                print(f"Error loading U-Net: {e}")
                MODEL = None
    
    # Build untrained model if no trained model found
    if MODEL is None:
        print("\n[WARNING] No trained model found.")
        print("Building untrained VGG-UNet model for testing...")
        MODEL = load_model(model_type='vgg_unet')
        MODEL_TYPE = 'vgg_unet'
        print("  [WARNING] Using untrained model - results will not be accurate!")
        print("  Please train a model first or place a .h5 file in the models/ directory.")
    
    # Warm up the model with a dummy prediction (first inference is slower)
    print("\nWarming up model...")
    dummy_input = np.zeros((1, 256, 256, 1), dtype=np.float32)
    try:
        _ = MODEL(dummy_input, training=False)
        print("[OK] Model ready for inference\n")
    except Exception as e:
        print(f"[WARNING] Model warmup failed: {e}\n")

# Load model on startup
load_colorization_model()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_array):
    """Convert numpy image array to base64 string."""
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Encode image to JPEG
    success, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise ValueError("Failed to encode image")
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"


@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/colorize', methods=['POST'])
def colorize_image():
    """
    Colorize a black-and-white image.
    
    Request:
        - Form data with 'image' file field
    
    Response:
        - JSON with 'colorized' (base64 image) and 'success' status
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Check file extension
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Read image file
        file_bytes = file.read()
        
        # Convert to numpy array using OpenCV
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                'success': False,
                'error': 'Could not decode image file'
            }), 400
        
        # Preprocess image for model input
        try:
            preprocessed = preprocess_image(img, is_file_path=False)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Image preprocessing failed: {str(e)}'
            }), 400
        
        # Run colorization
        try:
            import time
            start_time = time.time()
            colorized = predict_colorization(MODEL, preprocessed)
            elapsed_time = time.time() - start_time
            print(f"Colorization completed in {elapsed_time:.2f} seconds")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Colorization error: {error_details}")
            return jsonify({
                'success': False,
                'error': f'Colorization failed: {str(e)}'
            }), 500
        
        # Convert colorized image to base64
        try:
            colorized_base64 = image_to_base64(colorized)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Image encoding failed: {str(e)}'
            }), 500
        
        return jsonify({
            'success': True,
            'colorized': colorized_base64,
            'model_type': MODEL_TYPE
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'model_type': MODEL_TYPE
    })


if __name__ == '__main__':
    print("Starting Flask application...")
    print(f"Model loaded: {MODEL is not None}")
    print(f"Model type: {MODEL_TYPE}")
    app.run(debug=True, host='0.0.0.0', port=5000)

