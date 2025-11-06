"""Streamlit web UI for EdgeTwin."""

import streamlit as st
from pathlib import Path


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="EdgeTwin",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 EdgeTwin")
    st.markdown("**Hardware-aware co-simulation platform for robotics AI**")
    
    st.sidebar.header("Configuration")
    
    # Model upload
    uploaded_model = st.sidebar.file_uploader(
        "Upload ONNX Model",
        type=["onnx"]
    )
    
    # Video upload
    uploaded_video = st.sidebar.file_uploader(
        "Upload Input Video",
        type=["mp4", "avi", "mov"]
    )
    
    # SKU selection
    sku = st.sidebar.selectbox(
        "Jetson SKU",
        ["orin_super", "orin_nx", "orin_nano", "xavier_nx", "nano"]
    )
    
    # Configuration knobs
    st.sidebar.subheader("Model Configuration")
    precision = st.sidebar.selectbox("Precision", ["INT8", "FP16", "FP32"])
    resolution_h = st.sidebar.slider("Height", 320, 1280, 640, 32)
    resolution_w = st.sidebar.slider("Width", 320, 1280, 480, 32)
    batch_size = st.sidebar.slider("Batch Size", 1, 8, 1)
    frame_skip = st.sidebar.slider("Frame Skip", 0, 4, 0)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Profile")
        if st.button("Run Profile", type="primary"):
            st.info("Profiling not yet implemented. This will run Jetson profiling.")
    
    with col2:
        st.header("Predict")
        if st.button("Run Prediction"):
            st.info("Prediction not yet implemented. This will predict performance.")
    
    st.header("Results")
    st.info("Results will appear here after running profile/predict/optimize.")
    
    # Footer
    st.markdown("---")
    st.markdown("EdgeTwin v0.1.0 | Co-simulation platform for robotics AI")


if __name__ == "__main__":
    main()

