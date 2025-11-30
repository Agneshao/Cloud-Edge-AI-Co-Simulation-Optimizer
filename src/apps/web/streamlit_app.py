"""Streamlit web UI for EdgeTwin."""

import streamlit as st
from pathlib import Path
from src.core.cosim import ToyEnv, run_cosim, evaluate_logs


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
    
    # Co-Simulation Configuration
    st.sidebar.subheader("Co-Simulation")
    base_latency_ms = st.sidebar.slider("Base Latency (ms)", 10, 200, 50, 5)
    max_steps = st.sidebar.slider("Max Steps", 50, 500, 300, 50)
    
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
    
    # Co-Simulation Section
    st.header("🎯 EdgeTwin Co-Simulation")
    st.markdown(
        "Run hardware-aware co-simulation to see how latency affects task performance. "
        "The simulation demonstrates how hardware delays cause control failures."
    )
    
    if st.button("Run EdgeTwin Co-Sim", type="primary"):
        with st.spinner("Running co-simulation..."):
            # Create environment
            env = ToyEnv()
            
            # Run co-simulation
            logs = run_cosim(
                env=env,
                steps=max_steps,
                base_latency_ms=base_latency_ms,
                sku=sku,
                precision=precision,
                resolution=(resolution_h, resolution_w),
                batch_size=batch_size,
            )
            
            # Evaluate results
            df, collided = evaluate_logs(logs)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Steps Completed", len(logs))
            
            with col2:
                st.metric("Collision", "❌ YES" if collided else "✅ NO")
            
            with col3:
                max_offset = df["offset"].abs().max()
                st.metric("Max Offset", f"{max_offset:.3f}")
            
            # Visualization
            st.subheader("Offset Over Time")
            st.line_chart(df.set_index("step")["offset"])
            
            # Latency visualization
            st.subheader("Latency Over Time")
            st.line_chart(df.set_index("step")["latency_ms"])
            
            # Detailed metrics
            with st.expander("Detailed Metrics"):
                st.dataframe(df)
    
    st.header("Results")
    st.info("Results will appear here after running profile/predict/optimize.")
    
    # Footer
    st.markdown("---")
    st.markdown("EdgeTwin v0.1.0 | Co-simulation platform for robotics AI")


if __name__ == "__main__":
    main()

