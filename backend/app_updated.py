"""
TruthLens AI - Flask Backend API
Updated to work with properly trained model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import sys

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from model_handler import DeepfakeDetectorModel, load_model, predict_image, predict_video

# ============================================================
# FLASK APP SETUP
# ============================================================
app = Flask(__name__)
CORS(app)

# Config
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# ============================================================
# DEVICE & MODEL
# ============================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {DEVICE}")

# Load model
try:
    MODEL = load_model(device=DEVICE)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    MODEL = None

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def allowed_file(filename, file_type):
    """Check if file extension is allowed"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    if file_type == 'image':
        return ext in ALLOWED_IMAGE_EXTENSIONS
    elif file_type == 'video':
        return ext in ALLOWED_VIDEO_EXTENSIONS
    
    return False

# ============================================================
# ROUTES
# ============================================================

@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'TruthLens AI - Deepfake Detection API',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE)
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'model_ready': MODEL is not None
    })

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Detect if uploaded image is real or fake"""
    
    if MODEL is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    # Check if file exists
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Validate file
    if not allowed_file(file.filename, 'image'):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}'
        }), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        label, confidence, explanation = predict_image(filepath, MODEL, device=DEVICE)
        
        # Prepare response
        response = {
            'success': True,
            'filename': filename,
            'prediction': label,
            'confidence': float(confidence),
            'explanation': explanation,
            'is_deepfake': label == 'Fake'
        }
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error in detect_image: {e}")
        return jsonify({
            'success': False,
            'error': f'Error during analysis: {str(e)}'
        }), 500

@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    """Detect if uploaded video is real or fake"""
    
    if MODEL is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    # Check if file exists
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    # Validate file
    if not allowed_file(file.filename, 'video'):
        return jsonify({
            'success': False,
            'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}'
        }), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        label, confidence, explanation = predict_video(filepath, MODEL, device=DEVICE, frames_to_sample=10)
        
        # Prepare response
        response = {
            'success': True,
            'filename': filename,
            'prediction': label,
            'confidence': float(confidence),
            'explanation': explanation,
            'is_deepfake': label == 'Fake'
        }
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error in detect_video: {e}")
        return jsonify({
            'success': False,
            'error': f'Error during analysis: {str(e)}'
        }), 500

@app.route('/api/detect/url', methods=['POST'])
def detect_url():
    """Detect from image URL"""
    
    if MODEL is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({
            'success': False,
            'error': 'No URL provided'
        }), 400
    
    try:
        url = data['url']
        
        # Download image
        import urllib.request
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_url_image.jpg')
        urllib.request.urlretrieve(url, filename)
        
        # Make prediction
        label, confidence, explanation = predict_image(filename, MODEL, device=DEVICE)
        
        # Prepare response
        response = {
            'success': True,
            'url': url,
            'prediction': label,
            'confidence': float(confidence),
            'explanation': explanation,
            'is_deepfake': label == 'Fake'
        }
        
        # Clean up
        try:
            os.remove(filename)
        except:
            pass
        
        return jsonify(response)
    
    except Exception as e:
        print(f"❌ Error in detect_url: {e}")
        return jsonify({
            'success': False,
            'error': f'Error: {str(e)}'
        }), 500

@app.route('/api/info', methods=['GET'])
def info():
    """Get API information"""
    return jsonify({
        'name': 'TruthLens AI - Deepfake Detection',
        'version': '2.0',
        'device': str(DEVICE),
        'model_loaded': MODEL is not None,
        'endpoints': {
            'POST /api/detect/image': 'Detect if image is real or fake',
            'POST /api/detect/video': 'Detect if video is real or fake',
            'POST /api/detect/url': 'Detect from image URL'
        },
        'supported_formats': {
            'images': list(ALLOWED_IMAGE_EXTENSIONS),
            'videos': list(ALLOWED_VIDEO_EXTENSIONS)
        }
    })

# ============================================================
# ERROR HANDLERS
# ============================================================
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🚀 TruthLens AI - Deepfake Detection API")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {'✅ Loaded' if MODEL else '❌ Not loaded'}")
    print("=" * 60)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
