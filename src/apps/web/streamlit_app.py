"""Streamlit web UI for EdgeTwin."""

import streamlit as st
from pathlib import Path
import sys
import json
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.profile.pipeline_profiler import PipelineProfiler
from src.core.predict.latency_rule import predict_latency
from src.core.predict.power import predict_power
from src.core.predict.thermal_rc import ThermalRC
from src.core.optimize.knobs import ConfigKnobs
from src.core.optimize.search import greedy_search
from src.core.optimize.model_converter import optimize_model_for_metrics
from src.core.plan.reporter import ReportGenerator


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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Profile", "Predict", "Optimize", "Optimize Model", "Full Workflow"])
    
    with tab1:
        st.header("Profile Model")
        if st.button("Run Profile", type="primary"):
            if not uploaded_model:
                st.error("Please upload an ONNX model first")
            else:
                with st.spinner("Profiling model..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_model:
                        tmp_model.write(uploaded_model.getvalue())
                        tmp_model_path = tmp_model.name
                    
                    video_path = None
                    if uploaded_video:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                            tmp_video.write(uploaded_video.getvalue())
                            video_path = tmp_video.name
                    
                    try:
                        profiler = PipelineProfiler(sku=sku)
                        results = profiler.profile(
                            model_path=tmp_model_path,
                            video_path=video_path,
                            iterations=10
                        )
                        
                        st.success("Profiling complete!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Latency", f"{results['latency_ms']['total']:.2f} ms")
                        with col2:
                            st.metric("FPS", f"{results['fps']:.1f}")
                        with col3:
                            st.metric("Power", f"{results['power_w']:.2f} W")
                        with col4:
                            st.metric("Memory", f"{results['memory_mb']:.0f} MB")
                        
                        st.subheader("Stage Breakdown")
                        st.json({
                            "preprocess": f"{results['latency_ms']['preprocess']:.2f} ms",
                            "inference": f"{results['latency_ms']['inference']:.2f} ms",
                            "postprocess": f"{results['latency_ms']['postprocess']:.2f} ms",
                        })
                        
                        # Store in session state
                        st.session_state['profile_results'] = results
                    except Exception as e:
                        st.error(f"Profiling failed: {e}")
    
    with tab2:
        st.header("Predict Performance")
        
        if 'profile_results' not in st.session_state:
            st.info("Run profiling first to get baseline metrics")
        else:
            profile_results = st.session_state['profile_results']
            base_latency = profile_results['latency_ms']['total']
            base_power = profile_results['power_w']
            
            st.subheader("Configuration to Predict")
            pred_precision = st.selectbox("Precision", ["INT8", "FP16", "FP32"], key="pred_precision")
            pred_res_h = st.slider("Height", 320, 1280, 640, 32, key="pred_h")
            pred_res_w = st.slider("Width", 320, 1280, 480, 32, key="pred_w")
            pred_batch = st.slider("Batch Size", 1, 8, 1, key="pred_batch")
            
            if st.button("Run Prediction"):
                with st.spinner("Predicting..."):
                    pred_latency = predict_latency(
                        base_latency_ms=base_latency,
                        sku=sku,
                        precision=pred_precision,
                        resolution=(pred_res_h, pred_res_w),
                        batch_size=pred_batch
                    )
                    
                    pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 0
                    pred_power = predict_power(
                        base_power_w=base_power,
                        fps=pred_fps,
                        precision=pred_precision,
                        sku=sku,
                        resolution=(pred_res_h, pred_res_w)
                    )
                    
                    # Thermal prediction
                    thermal_model = ThermalRC(
                        ambient_temp_c=25.0,
                        thermal_resistance_c_per_w=0.5,
                        thermal_capacitance_j_per_c=10.0,
                        max_temp_c=70.0
                    )
                    time_to_throttle = thermal_model.time_to_throttle(pred_power)
                    
                    st.success("Prediction complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Latency", f"{pred_latency:.2f} ms")
                    with col2:
                        st.metric("Predicted Power", f"{pred_power:.2f} W")
                    with col3:
                        st.metric("Predicted FPS", f"{pred_fps:.1f}")
                    
                    if time_to_throttle != float('inf'):
                        st.metric("Time to Throttle", f"{time_to_throttle:.1f} s")
                    else:
                        st.metric("Time to Throttle", "Never")
    
    with tab3:
        st.header("Optimize Configuration")
        
        if 'profile_results' not in st.session_state:
            st.info("Run profiling first to get baseline metrics")
        else:
            profile_results = st.session_state['profile_results']
            
            st.subheader("Constraints")
            max_power = st.slider("Max Power (W)", 5.0, 30.0, 20.0, 0.5)
            max_latency = st.slider("Max Latency (ms)", 10.0, 200.0, 100.0, 5.0)
            
            st.subheader("Objective Weights")
            weight_latency = st.slider("Latency Weight", 0.0, 2.0, 1.0, 0.1)
            weight_power = st.slider("Power Weight", 0.0, 2.0, 0.1, 0.1)
            
            if st.button("Run Optimization", type="primary"):
                with st.spinner("Optimizing configuration..."):
                    base_latency = profile_results['latency_ms']['total']
                    base_power = profile_results['power_w']
                    
                    def objective(knobs: ConfigKnobs) -> float:
                        pred_latency = predict_latency(
                            base_latency, sku, knobs.precision,
                            knobs.resolution, knobs.batch_size
                        )
                        pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 30.0
                        pred_power = predict_power(
                            base_power, pred_fps, knobs.precision,
                            sku, knobs.resolution
                        )
                        
                        if pred_power > max_power or pred_latency > max_latency:
                            return 10000.0
                        
                        return weight_latency * pred_latency + weight_power * pred_power
                    
                    best_knobs = greedy_search(
                        objective_fn=objective,
                        initial_knobs=ConfigKnobs(),
                        max_iterations=30
                    )
                    
                    # Get predictions
                    best_latency = predict_latency(
                        base_latency, sku, best_knobs.precision,
                        best_knobs.resolution, best_knobs.batch_size
                    )
                    best_fps = 1000.0 / best_latency if best_latency > 0 else 0
                    best_power = predict_power(
                        base_power, best_fps, best_knobs.precision,
                        sku, best_knobs.resolution
                    )
                    
                    st.success("Optimization complete!")
                    
                    st.subheader("Best Configuration")
                    st.json(best_knobs.to_dict())
                    
                    st.subheader("Predicted Performance")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Latency", f"{best_latency:.2f} ms")
                    with col2:
                        st.metric("Power", f"{best_power:.2f} W")
                    with col3:
                        st.metric("FPS", f"{best_fps:.1f}")
                    
                    st.session_state['best_knobs'] = best_knobs
                    st.session_state['best_performance'] = {
                        "latency_ms": best_latency,
                        "power_w": best_power,
                        "fps": best_fps
                    }
    
    with tab4:
        st.header("Optimize Model File")
        st.info("Upload a model and specify target metrics to get an optimized model file")
        
        if not uploaded_model:
            st.warning("Please upload an ONNX model first")
        else:
            st.subheader("Target Metrics")
            col1, col2 = st.columns(2)
            
            with col1:
                target_fps = st.number_input(
                    "Target FPS",
                    min_value=1.0,
                    max_value=120.0,
                    value=30.0,
                    step=1.0,
                    help="Target frames per second to achieve"
                )
                target_latency_ms = st.number_input(
                    "Target Latency (ms)",
                    min_value=1.0,
                    max_value=1000.0,
                    value=33.33,
                    step=1.0,
                    help="Target latency in milliseconds (alternative to FPS)"
                )
            
            with col2:
                max_power_w = st.number_input(
                    "Max Power (W)",
                    min_value=1.0,
                    max_value=50.0,
                    value=20.0,
                    step=0.5,
                    help="Maximum power constraint"
                )
            
            if st.button("Optimize Model", type="primary"):
                with st.spinner("Optimizing model..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_model:
                        tmp_model.write(uploaded_model.getvalue())
                        tmp_model_path = tmp_model.name
                    
                    # Create output path
                    output_path = tmp_model_path.replace(".onnx", "_optimized.onnx")
                    
                    try:
                        results = optimize_model_for_metrics(
                            model_path=tmp_model_path,
                            target_fps=target_fps if target_fps > 0 else None,
                            target_latency_ms=target_latency_ms if target_latency_ms > 0 else None,
                            max_power_w=max_power_w if max_power_w > 0 else None,
                            sku=sku,
                            output_path=output_path
                        )
                        
                        st.success("Model optimization complete!")
                        
                        st.subheader("Optimization Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Achieved FPS", f"{results['achieved_fps']:.1f}")
                        with col2:
                            st.metric("Latency", f"{results['latency_ms']:.2f} ms")
                        with col3:
                            st.metric("Power", f"{results['power_w']:.2f} W")
                        
                        st.write(f"**Precision:** {results['precision']}")
                        st.write(f"**Resolution:** {results['resolution']}")
                        st.write(f"**Status:** {results['message']}")
                        
                        # Download button for optimized model
                        if results['optimization_applied'] and Path(results['optimized_model_path']).exists():
                            with open(results['optimized_model_path'], 'rb') as f:
                                st.download_button(
                                    label="Download Optimized Model",
                                    data=f.read(),
                                    file_name=f"optimized_{uploaded_model.name}",
                                    mime="application/octet-stream"
                                )
                        else:
                            st.info("Model optimization was not applied. See message above.")
                    
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
    
    with tab5:
        st.header("Full Workflow")
        st.info("Run complete workflow: Profile → Predict → Optimize → Report")
        
        if st.button("Run Full Workflow", type="primary"):
            if not uploaded_model:
                st.error("Please upload an ONNX model first")
            else:
                progress = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Profile
                status_text.text("Step 1/4: Profiling...")
                progress.progress(25)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp_model:
                    tmp_model.write(uploaded_model.getvalue())
                    tmp_model_path = tmp_model.name
                
                video_path = None
                if uploaded_video:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                        tmp_video.write(uploaded_video.getvalue())
                        video_path = tmp_video.name
                
                profiler = PipelineProfiler(sku=sku)
                profile_results = profiler.profile(
                    model_path=tmp_model_path,
                    video_path=video_path,
                    iterations=10
                )
                
                # Step 2: Predict
                status_text.text("Step 2/4: Predicting...")
                progress.progress(50)
                
                base_latency = profile_results['latency_ms']['total']
                base_power = profile_results['power_w']
                
                # Step 3: Optimize
                status_text.text("Step 3/4: Optimizing...")
                progress.progress(75)
                
                def objective(knobs: ConfigKnobs) -> float:
                    pred_latency = predict_latency(
                        base_latency, sku, knobs.precision,
                        knobs.resolution, knobs.batch_size
                    )
                    pred_fps = 1000.0 / pred_latency if pred_latency > 0 else 30.0
                    pred_power = predict_power(
                        base_power, pred_fps, knobs.precision,
                        sku, knobs.resolution
                    )
                    if pred_power > 20.0:
                        return 10000.0
                    return pred_latency + 0.1 * pred_power
                
                best_knobs = greedy_search(objective_fn=objective, max_iterations=20)
                
                best_latency = predict_latency(
                    base_latency, sku, best_knobs.precision,
                    best_knobs.resolution, best_knobs.batch_size
                )
                best_fps = 1000.0 / best_latency if best_latency > 0 else 0
                best_power = predict_power(
                    base_power, best_fps, best_knobs.precision,
                    sku, best_knobs.resolution
                )
                
                # Step 4: Report
                status_text.text("Step 4/4: Generating report...")
                progress.progress(100)
                
                report_gen = ReportGenerator()
                report_path = report_gen.generate_report(
                    profile_results=profile_results,
                    predictions={
                        "latency_ms": best_latency,
                        "power_w": best_power,
                    },
                    optimized_config=best_knobs.to_dict()
                )
                
                status_text.text("Complete!")
                st.success("Full workflow complete!")
                
                st.subheader("Results Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Baseline:**")
                    st.write(f"- Latency: {base_latency:.2f} ms")
                    st.write(f"- Power: {base_power:.2f} W")
    with col2:
                    st.write("**Optimized:**")
                    st.write(f"- Latency: {best_latency:.2f} ms")
                    st.write(f"- Power: {best_power:.2f} W")
                    st.write(f"- Config: {best_knobs.precision} @ {best_knobs.resolution}")
                
                st.download_button(
                    label="Download Report",
                    data=Path(report_path).read_text(),
                    file_name="edgetwin_report.html",
                    mime="text/html"
                )
    
    # Results section (always visible)
    if 'profile_results' in st.session_state or 'best_knobs' in st.session_state:
    st.header("Results")
        if 'profile_results' in st.session_state:
            st.json(st.session_state['profile_results'])
        if 'best_knobs' in st.session_state:
            st.json(st.session_state['best_knobs'].to_dict())
    
    # Footer
    st.markdown("---")
    st.markdown("EdgeTwin v0.1.0 | Co-simulation platform for robotics AI")


if __name__ == "__main__":
    main()

