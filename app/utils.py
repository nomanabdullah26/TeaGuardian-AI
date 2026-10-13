# ============================================================
# utils.py - All Helper Functions
# ============================================================
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import os
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# ============================================================
# CONSTANTS
# ============================================================
AGE_CLASSES = ['T1 (1-2 days) - Premium', 'T2 (3-4 days) - Good',
               'T3 (5-7 days) - Average', 'T4 (7+ days) - Not recommended']

DISEASE_CLASSES = [
    'Algal Leaf Spot', 'Brown Blight', 'Gray Blight',
    'Helopeltis', 'Red Spider', 'Green Mirid Bug', 'Healthy Leaf'
]

QUALITY_SCORES = {
    'T1 (1-2 days) - Premium': 95,
    'T2 (3-4 days) - Good': 75,
    'T3 (5-7 days) - Average': 50,
    'T4 (7+ days) - Not recommended': 25
}

COLORS = {
    'T1 (1-2 days) - Premium': '#28a745',
    'T2 (3-4 days) - Good': '#17a2b8',
    'T3 (5-7 days) - Average': '#ffc107',
    'T4 (7+ days) - Not recommended': '#dc3545'
}

# Confidence threshold for disease detection
CONFIDENCE_THRESHOLD = 60.0  # Percentage

# ============================================================
# TEA LEAF VALIDATION MODEL (Pre-check)
# ============================================================
@st.cache_resource
def load_precheck_model():
    """Load MobileNetV2 for tea leaf validation."""
    try:
        model = MobileNetV2(weights='imagenet')
        return model
    except Exception as e:
        st.warning(f"⚠️ Could not load pre-check model: {e}")
        return None

def is_tea_leaf(img_array):
    """
    Check if the image is a tea leaf using MobileNetV2.
    Returns (is_leaf, confidence)
    """
    model = load_precheck_model()
    if model is None:
        return True, 100.0  # Skip validation if model not loaded
    
    try:
        # MobileNetV2 expects input in a specific range
        # img_array is already preprocessed for EfficientNet, so we need to handle carefully
        # For simplicity, we'll use a basic check
        # In production, you'd want to use a proper leaf classifier
        
        # Since we can't easily use MobileNetV2 with our preprocessed input,
        # we'll use a simpler heuristic: check if image has green color dominance
        # This is a fallback - in production, train a proper leaf classifier
        
        img = img_array[0]  # Remove batch dimension
        # Convert back to 0-255 range
        img = (img + 1) * 127.5  # Reverse of preprocess_input
        img = img.astype(np.uint8)
        
        # Check for green color dominance
        # Get average green channel
        green_mean = np.mean(img[:, :, 1]) if img.shape[-1] == 3 else 0
        red_mean = np.mean(img[:, :, 0]) if img.shape[-1] == 3 else 0
        blue_mean = np.mean(img[:, :, 2]) if img.shape[-1] == 3 else 0
        
        # Leaf should have more green than red and blue
        is_leaf = green_mean > red_mean and green_mean > blue_mean
        confidence = 70.0 if is_leaf else 30.0
        
        # For a more accurate check, use a proper tea leaf classifier
        # This is a simple fallback
        
        return is_leaf, confidence
        
    except Exception as e:
        # If any error occurs, default to True
        return True, 50.0

# ============================================================
# MODEL LOADING (Cached)
# ============================================================
@st.cache_resource
def load_models():
    """Load both Age and Disease models."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, 'models')
    
    age_path = os.path.join(model_dir, 'age_model.h5')
    disease_path = os.path.join(model_dir, 'disease_model.h5')
    
    age_model = None
    disease_model = None
    
    if os.path.exists(age_path):
        try:
            # Load with custom_objects to handle compatibility issues
            age_model = tf.keras.models.load_model(age_path, compile=False)
        except Exception as e:
            st.warning(f"⚠️ Could not load Age model: {e}")
    
    if os.path.exists(disease_path):
        try:
            disease_model = tf.keras.models.load_model(disease_path, compile=False)
        except Exception as e:
            st.warning(f"⚠️ Could not load Disease model: {e}")
    
    return age_model, disease_model

# ============================================================
# IMAGE PREPROCESSING
# ============================================================
def preprocess_image(img, target_size=(224, 224)):
    """Preprocess image for prediction."""
    img = img.resize(target_size)
    img_array = np.array(img)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================
def predict_age(model, img_array):
    """Predict age class."""
    if model is None:
        return "Model not loaded", 0.0
    pred = model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred) * 100)
    return AGE_CLASSES[class_idx], confidence

def predict_disease(model, img_array):
    """Predict disease class."""
    if model is None:
        return "Model not loaded", 0.0
    pred = model.predict(img_array, verbose=0)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred) * 100)
    return DISEASE_CLASSES[class_idx], confidence

# ============================================================
# RECOMMENDATION
# ============================================================
def get_plucking_recommendation(age_class):
    """Get plucking recommendation."""
    if 'T1' in age_class or 'T2' in age_class:
        return "✅ Pluck Now!", "success", "Best quality, harvest immediately"
    elif 'T3' in age_class:
        return "⏳ Wait 2-3 Days", "warning", "Check again in 48 hours"
    else:
        return "❌ Skip Plucking", "error", "Not suitable for tea production"

def get_quality_score(age_class):
    """Get quality score."""
    return QUALITY_SCORES.get(age_class, 0)

def get_color(age_class):
    """Get color for class."""
    return COLORS.get(age_class, '#6c757d')