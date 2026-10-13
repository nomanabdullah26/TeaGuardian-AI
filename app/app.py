# ============================================================
# app.py - TeaGuardian Main Application
# ============================================================
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import time
import os
import pandas as pd
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

from utils import (
    load_models,
    preprocess_image,
    predict_age,
    predict_disease,
    get_plucking_recommendation,
    get_quality_score,
    get_color,
    AGE_CLASSES,
    DISEASE_CLASSES,
    CONFIDENCE_THRESHOLD,
    is_tea_leaf
)

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="TeaGuardian AI 🍃",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD CSS
# ============================================================
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), 'styles.css')
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div style="display: flex; justify-content: center; align-items: center; gap: 15px; padding: 20px 0;">
    <span style="font-size: 4rem; animation: float 2s ease-in-out infinite;">🍃</span>
    <span style="font-size: 3rem; animation: spin 4s linear infinite;">☕</span>
    <span style="font-size: 4rem; animation: float 2s ease-in-out infinite 0.5s;">🍃</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style="text-align: center; color: #1a3a1a; font-weight: 800;">
        🍃 TeaGuardian AI
    </h1>
    <p style="text-align: center; font-size: 1.2rem; color: #1a3a1a;">
        AI-Powered Tea Leaf Quality & Disease Detection System
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🍵 TeaGuardian AI")
    st.markdown("---")
    st.markdown("""
    **AI-Powered Tea Leaf Analyzer**
    
    - 🍃 **Quality Grading** (T1-T4)
    - 🦠 **Disease Detection** (7 Classes)
    - ⏰ **Plucking Recommendations**
    
    ---
    **How it Works:**
    1. 📸 Upload a tea leaf image
    2. 🧠 AI analyzes the leaf
    3. 📊 Get instant results!
    
    ---
    **Tech Stack:**
    - EfficientNetB0 (Age)
    - DenseNet201 (Disease)
    - Streamlit
    - Plotly
    """)
    st.markdown("---")
    st.caption("🚀 Developed by Afrin Binte Amin")

# ============================================================
# LOAD MODELS
# ============================================================
with st.spinner("🧠 Loading AI Models..."):
    age_model, disease_model = load_models()

if age_model is None and disease_model is None:
    st.error("❌ No models found! Please place model files in the 'models/' folder.")
    st.stop()

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Image Analysis",
    "📊 Quality Guide",
    "📈 Data Insights",
    "ℹ️ About"
])

# ============================================================
# TAB 1: IMAGE ANALYSIS
# ============================================================
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose a tea leaf image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a tea leaf"
        )
        
        # ============================================================
        # CAMERA WITH ON/OFF CONTROL
        # ============================================================
        st.markdown("### 📷 Camera")
        
        # Camera state
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
        
        # Camera toggle button
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            if st.button("📷 Start Camera", type="primary" if not st.session_state.camera_on else "secondary"):
                st.session_state.camera_on = True
        with col_cam2:
            if st.button("⏹️ Stop Camera", type="primary" if st.session_state.camera_on else "secondary"):
                st.session_state.camera_on = False
        
        # Camera display
        if st.session_state.camera_on:
            camera_image = st.camera_input(
                "📷 Take a photo",
                help="Click to capture image from camera"
            )
            if camera_image:
                uploaded_file = camera_image
        else:
            st.info("👆 Click 'Start Camera' to enable camera")
        
        # Analysis Type
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["🍃 Quality (Age)", "🦠 Disease", "🔬 Both"],
            horizontal=True
        )
    
    if uploaded_file is not None:
        with col1:
            img = Image.open(uploaded_file)
            st.image(img, caption="📷 Uploaded Tea Leaf", use_column_width=True)
        
        with col2:
            st.markdown("### 🔍 Analysis Results")
            
            if st.button("🚀 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing..."):
                    time.sleep(0.5)
                    
                    img_array = preprocess_image(img)
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    # ============================================================
                    # TEA LEAF VALIDATION CHECK
                    # ============================================================
                    is_leaf, leaf_confidence = is_tea_leaf(img_array)
                    
                    if not is_leaf:
                        st.warning(f"⚠️ This doesn't appear to be a tea leaf image (Confidence: {leaf_confidence:.1f}%)")
                        st.info("📋 Please upload a clear image of a tea leaf for accurate analysis.")
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.stop()
                    
                    # ============================================================
                    # AGE ANALYSIS
                    # ============================================================
                    if analysis_type in ["🍃 Quality (Age)", "🔬 Both"]:
                        st.markdown("#### 🍃 Quality Analysis")
                        
                        if age_model is not None:
                            age_class, age_conf = predict_age(age_model, img_array)
                            color = get_color(age_class)
                            quality_score = get_quality_score(age_class)
                            recommendation, status, detail = get_plucking_recommendation(age_class)
                            
                            # Quality Gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=quality_score,
                                title={'text': "Quality Score"},
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': color},
                                    'steps': [
                                        {'range': [0, 30], 'color': "red"},
                                        {'range': [30, 60], 'color': "yellow"},
                                        {'range': [60, 80], 'color': "lightgreen"},
                                        {'range': [80, 100], 'color': "darkgreen"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Prediction
                            st.markdown(f"""
                            <div style="background: {color}; padding: 15px; border-radius: 10px; text-align: center; color: white; margin: 10px 0;">
                                <h3 style="margin: 0;">🍃 {age_class}</h3>
                                <p style="margin: 5px 0 0 0;">Confidence: {age_conf:.2f}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Recommendation
                            status_color = "#28a745" if "success" in status else "#ffc107" if "warning" in status else "#dc3545"
                            st.markdown(f"""
                            <div style="background: {status_color}; padding: 12px; border-radius: 10px; text-align: center; color: white;">
                                <h4 style="margin: 0;">{recommendation}</h4>
                                <p style="margin: 5px 0 0 0;">{detail}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Confidence Bar
                            fig2 = go.Figure(go.Bar(
                                x=['Confidence'],
                                y=[age_conf],
                                text=[f'{age_conf:.1f}%'],
                                textposition='outside',
                                marker_color=color
                            ))
                            fig2.update_layout(yaxis_range=[0, 100], height=150, showlegend=False)
                            st.plotly_chart(fig2, use_container_width=True)
                        else:
                            st.warning("⚠️ Age model not loaded")
                    
                    # ============================================================
                    # DISEASE ANALYSIS
                    # ============================================================
                    if analysis_type in ["🦠 Disease", "🔬 Both"]:
                        st.markdown("#### 🦠 Disease Detection")
                        
                        if disease_model is not None:
                            disease_class, disease_conf = predict_disease(disease_model, img_array)
                            
                            # Check if confidence is below threshold
                            if disease_conf < CONFIDENCE_THRESHOLD:
                                color = "#ffc107"  # Warning color
                                emoji = "❓"
                                st.markdown(f"""
                                <div style="background: {color}; padding: 15px; border-radius: 10px; text-align: center; color: white; margin: 10px 0;">
                                    <h3 style="margin: 0;">{emoji} Unknown Disease Pattern</h3>
                                    <p style="margin: 5px 0 0 0;">Confidence: {disease_conf:.2f}% (Below threshold)</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                st.warning("⚠️ This disease pattern is not recognized in our database with high confidence.")
                                
                                # Show known diseases list
                                with st.expander("📋 Known Diseases in Database", expanded=True):
                                    st.write("Our model is trained on these 7 classes:")
                                    for d in DISEASE_CLASSES:
                                        st.write(f"- {d}")
                                
                            else:
                                color = "#28a745" if "Healthy" in disease_class else "#dc3545"
                                emoji = "🌿" if "Healthy" in disease_class else "🦠"
                                
                                st.markdown(f"""
                                <div style="background: {color}; padding: 15px; border-radius: 10px; text-align: center; color: white; margin: 10px 0;">
                                    <h3 style="margin: 0;">{emoji} {disease_class}</h3>
                                    <p style="margin: 5px 0 0 0;">Confidence: {disease_conf:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if "Healthy" in disease_class:
                                    st.success("✅ Leaf is healthy!")
                                else:
                                    st.error("⚠️ Disease detected! Contact expert.")
                            
                            # Confidence Bar
                            bar_color = "#ffc107" if disease_conf < CONFIDENCE_THRESHOLD else color
                            fig3 = go.Figure(go.Bar(
                                x=['Confidence'],
                                y=[disease_conf],
                                text=[f'{disease_conf:.1f}%'],
                                textposition='outside',
                                marker_color=bar_color
                            ))
                            fig3.update_layout(yaxis_range=[0, 100], height=150, showlegend=False)
                            st.plotly_chart(fig3, use_container_width=True)
                        else:
                            st.warning("⚠️ Disease model not loaded")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# TAB 2: QUALITY GUIDE
# ============================================================
with tab2:
    st.markdown("### 🍃 Tea Leaf Quality Guide")
    
    quality_data = pd.DataFrame({
        'Grade': ['T1', 'T2', 'T3', 'T4'],
        'Name': ['Premium', 'Good', 'Average', 'Poor'],
        'Age': ['1-2 days', '3-4 days', '5-7 days', '7+ days'],
        'Score': [95, 75, 50, 25],
        'Action': ['Pluck Now!', 'Pluck Now!', 'Wait 2-3 Days', 'Skip Plucking'],
        'Color': ['#28a745', '#17a2b8', '#ffc107', '#dc3545'],
        'Emoji': ['🌟', '👍', '⚠️', '❌']
    })
    
    for _, row in quality_data.iterrows():
        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
        with col1:
            st.markdown(f"""
            <div style="background: {row['Color']}; padding: 15px; border-radius: 10px; text-align: center; color: white;">
                <h2>{row['Emoji']}</h2>
                <h3>{row['Grade']}</h3>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.write(f"**{row['Name']}**")
            st.write(f"📅 {row['Age']}")
        with col3:
            st.progress(row['Score'] / 100)
            st.write(f"{row['Score']}/100")
        with col4:
            st.write(f"📊 {row['Score']}%")
        with col5:
            if 'Pluck' in row['Action']:
                st.success(f"✅ {row['Action']}")
            elif 'Wait' in row['Action']:
                st.warning(f"⏳ {row['Action']}")
            else:
                st.error(f"❌ {row['Action']}")
        st.markdown("---")

# ============================================================
# TAB 3: DATA INSIGHTS
# ============================================================
with tab3:
    st.markdown("### 📊 Dataset & Model Statistics")
    
    # Model Statistics - Updated with correct values
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📸 Age Dataset", value="2,208", delta="4 Classes")
    with col2:
        st.metric(label="🦠 Disease Dataset", value="5,278", delta="7 Classes")
    with col3:
        st.metric(label="🏆 Age Accuracy", value="74%", delta="✅")
    with col4:
        st.metric(label="🏆 Disease Accuracy", value="~85%", delta="✅")
    
    # Summary Section
    st.markdown("---")
    st.markdown("### 📋 Model Performance Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background: #f0f7f0; padding: 20px; border-radius: 12px; border-left: 5px solid #2d5016;">
            <h4 style="color: #1a3a1a;">🍃 Age Model</h4>
            <p><strong>Training Accuracy:</strong> 76.2%</p>
            <p><strong>Validation Accuracy:</strong> 74.0%</p>
            <p><strong>Classes:</strong> 4 (T1-T4)</p>
            <p><strong>Architecture:</strong> EfficientNetB0</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: #f0f7f0; padding: 20px; border-radius: 12px; border-left: 5px solid #dc3545;">
            <h4 style="color: #1a3a1a;">🦠 Disease Model</h4>
            <p><strong>Training Accuracy:</strong> 95.1%</p>
            <p><strong>Validation Accuracy:</strong> 85.0%</p>
            <p><strong>Classes:</strong> 7</p>
            <p><strong>Architecture:</strong> DenseNet201</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Age Class Distribution
    st.markdown("#### 🍃 Age Class Distribution")
    age_dist = pd.DataFrame({
        'Class': ['T1', 'T2', 'T3', 'T4'],
        'Images': [562, 615, 508, 523]
    })
    fig = px.bar(age_dist, x='Class', y='Images', color='Class', title="Age Dataset Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Disease Class Distribution
    st.markdown("#### 🦠 Disease Class Distribution")
    disease_dist = pd.DataFrame({
        'Disease': ['Algal Spot', 'Brown Blight', 'Gray Blight', 'Helopeltis', 'Red Spider', 'Green Mirid', 'Healthy'],
        'Images': [418, 508, 1013, 607, 515, 1282, 935]
    })
    fig = px.pie(disease_dist, values='Images', names='Disease', title="Disease Dataset Distribution")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 4: ABOUT
# ============================================================
with tab4:
    st.markdown("""
    ### 🌱 About TeaGuardian
    
    **TeaGuardian** is an AI-powered system for tea leaf quality assessment and disease detection.
    
    ---
    
    ### 🎯 Features
    - 🍃 **Quality Grading:** 4 grades (T1-T4)
    - 🦠 **Disease Detection:** 7 classes
    - ⏰ **Plucking Recommendations**
    - 📊 **Real-time Analysis**
    
    ---
    
    ### 📍 Dataset
    - **Source:** Tea gardens in Sylhet, Sreemangal, Moulvibazar
    - **Age Dataset:** 2,208 images
    - **Disease Dataset:** 5,278 images
    
    ---
    
    ### 🏆 Models
    - **Age Model:** EfficientNetB0 (Transfer Learning) - 74% Accuracy
    - **Disease Model:** DenseNet201 (Transfer Learning) - 85% Accuracy
    
    ---
    
    ### 🛠️ Technology
    - TensorFlow 2.15
    - Streamlit 1.44.1
    - Plotly 5.18.0
    """)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #1a3a1a; padding: 20px 0;">
        🍃 TeaGuardian AI &bull; 2026
    </div>
""", unsafe_allow_html=True)

st.caption("🚀 Developed by Iostream")