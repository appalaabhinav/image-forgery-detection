"""
AI-Based Image Forgery Detection Interface (CG-1050)
===================================================
A modern, professional Streamlit web application for digital image forensics.
Powered by a custom deep residual CNN trained from scratch (zero pretrained weights).
"""

import io
import time
from pathlib import Path
from typing import Optional
from PIL import Image
import streamlit as st

from inference import ForgeryDetector, MODEL_REGISTRY
from preprocessing import load_image_safely, compute_image_metadata

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="AI-Based Image Forgery Detection | CG-1050 Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern academic UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header styling */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-scratch {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-cg1050 {
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
        margin-left: 0.5rem;
    }

    /* Section Cards */
    .card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }

    /* Result Banners */
    .result-banner {
        border-radius: 14px;
        padding: 1.75rem 2rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .result-tampered {
        background: linear-gradient(135deg, #b91c1c 0%, #dc2626 100%);
        border: 1px solid #f87171;
        box-shadow: 0 8px 20px -4px rgba(220, 38, 38, 0.35);
    }
    .result-authentic {
        background: linear-gradient(135deg, #15803d 0%, #16a34a 100%);
        border: 1px solid #4ade80;
        box-shadow: 0 8px 20px -4px rgba(22, 163, 74, 0.35);
    }

    .result-title {
        font-size: 1.85rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.01em;
    }
    .result-subtitle {
        font-size: 1.05rem;
        opacity: 0.92;
        font-weight: 400;
    }

    /* Metadata pills */
    .meta-tag {
        background-color: #f1f5f9;
        color: #334155;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 500;
        margin-right: 0.4rem;
        display: inline-block;
        margin-bottom: 0.4rem;
    }

    /* Disclaimer box */
    .forensic-disclaimer {
        background-color: #f8fafc;
        border-left: 4px solid #64748b;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #475569;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Cached Model Loader
# ==============================================================================
@st.cache_resource(show_spinner=False)
def get_detector(model_key: str) -> ForgeryDetector:
    """Loads and caches the model checkpoint once in memory."""
    return ForgeryDetector(model_key=model_key)


# ==============================================================================
# Sidebar Configuration & Controls
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Model Configuration")
    st.caption("Custom CNN Architecture Selection")

    model_options = {
        "dual_stream": "Custom Dual-Stream (RGB + ELA, Scratch) [Recommended]",
        "official_benchmark": "Official Benchmark Custom CNN (From Scratch)",
        "boosted_transfer": "Boosted Dual-Stream (High Accuracy 75%–82%)",
        "clahe": "Custom Single-Stream (Mild CLAHE, Scratch)",
        "baseline": "Custom Single-Stream (Baseline RGB, Scratch)"
    }

    selected_model_key = st.selectbox(
        "Active Model Architecture:",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=0,
        help="Select which model checkpoint to use for evaluation."
    )

    selected_config = MODEL_REGISTRY.get(selected_model_key, {})

    default_tau = float(selected_config.get("threshold", 0.50))
    threshold_slider = st.slider(
        "Forensic Decision Boundary (τ):",
        min_value=0.20,
        max_value=0.90,
        value=default_tau,
        step=0.01,
        help="Decision threshold for classifying an image as TAMPERED. Calibrated on validation data to optimize accuracy."
    )

    st.markdown("---")
    st.subheader("Model Specs & Provenance")
    st.markdown(f"**Architecture:** `{selected_config.get('name', selected_model_key)}`")
    st.markdown(f"**Weights:** `{'Trained From Scratch (Zero Pretrained)' if selected_config.get('is_scratch', True) else 'Fine-Tuned Transfer Backbones'}`")
    st.markdown(f"**Input Dimensions:** `{selected_config.get('input_dim', '128×128')}`")
    st.markdown(f"**Accuracy:** `{selected_config.get('test_accuracy', 'N/A')}`")
    st.markdown(f"**ROC-AUC:** `{selected_config.get('test_auc', 'N/A')}`")
    st.markdown(f"**Tampered Recall:** `{selected_config.get('test_recall', 'N/A')}`")
    st.markdown(f"**Default Calibration:** `τ* = {default_tau:.2f}`")
    st.markdown(f"**Active Threshold:** `τ = {threshold_slider:.2f}`")
    st.markdown(f"**Checkpoint:** `{selected_config.get('checkpoint', 'N/A')}`")

    st.markdown("---")
    st.subheader("Dataset Details")
    st.markdown("""
    * **Benchmark:** CG-1050 Forgery Dataset
    * **Splitting:** 0-Scene Leakage Grouped Partition
    * **Manipulations:** Copy-Move & Splicing
    * **Classes:** Class 0 = ORIGINAL, Class 1 = TAMPERED
    """)

    st.markdown("---")
    st.subheader("Quick Test Samples")
    st.caption("Click to load pre-verified sample images directly from CG-1050:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Authentic Sample", use_container_width=True):
            st.session_state["sample_image_path"] = "TRAINING_CG-1050/TRAINING/ORIGINAL/Im10_2.jpg"
            st.session_state["sample_type"] = "Authentic Camera Capture"
            st.session_state["uploaded_file_data"] = None
    with col_btn2:
        if st.button("Tampered Sample", use_container_width=True):
            st.session_state["sample_image_path"] = "TRAINING_CG-1050/TRAINING/TAMPERED/Im100_cm2.jpg"
            st.session_state["sample_type"] = "Copy-Move Forgery"
            st.session_state["uploaded_file_data"] = None


# ==============================================================================
# Header Section
# ==============================================================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🛡️ AI-Based Image Forgery Detection</span>
    </div>
    <div class="hero-subtitle">
        Deep Learning-Based Digital Image Forensics using From-Scratch Residual CNNs & Error Level Analysis (ELA)
    </div>
    <div>
        <span class="badge-pill badge-scratch">✓ From-Scratch CNN (Zero ImageNet Weights)</span>
        <span class="badge-pill badge-cg1050">CG-1050 Benchmark Certified</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Image Upload & Handling
# ==============================================================================
col_upload, col_preview = st.columns([1.1, 0.9], gap="large")

with col_upload:
    st.markdown("### 1. Select or Upload Image")
    st.write("Upload a color image (`.jpg`, `.jpeg`, `.png`) to inspect for tampering artifacts.")

    uploaded_file = st.file_uploader(
        "Choose an image file:",
        type=["jpg", "jpeg", "png"],
        help="Upload standard image files. Max file size: 25MB."
    )

    # Check if a sample was triggered from sidebar
    active_image = None
    active_filename = None
    active_bytes = None

    if uploaded_file is not None:
        try:
            active_bytes = uploaded_file.getvalue()
            active_image = load_image_safely(io.BytesIO(active_bytes))
            active_filename = uploaded_file.name
            st.session_state["sample_image_path"] = None  # clear sample
        except Exception as e:
            st.error(f"Error loading uploaded image: {str(e)}")

    elif st.session_state.get("sample_image_path"):
        sample_p = Path(st.session_state["sample_image_path"])
        if sample_p.exists():
            active_image = load_image_safely(str(sample_p))
            active_filename = f"{sample_p.name} ({st.session_state.get('sample_type', 'Sample')})"
            with open(sample_p, "rb") as f:
                active_bytes = f.read()
            st.info(f"Loaded Sample: **{active_filename}**")

    # Action buttons
    if active_image is not None:
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            analyze_clicked = st.button("🔍 Analyze Image", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("✖ Reset Image", use_container_width=True):
                st.session_state["sample_image_path"] = None
                st.session_state["uploaded_file_data"] = None
                st.session_state["analysis_results"] = None
                st.rerun()
    else:
        analyze_clicked = False
        st.info("Please upload an image above or click one of the quick test sample buttons in the sidebar.")


with col_preview:
    st.markdown("### 2. Image Inspection")
    if active_image is not None:
        st.image(active_image, caption=f"Active Target: {active_filename}", use_container_width=True)

        # Image Metadata Box
        meta = compute_image_metadata(active_image, active_bytes)
        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:0.8rem; margin-top:0.5rem;">
            <span class="meta-tag">📏 <b>Resolution:</b> {meta['dimensions']}</span>
            <span class="meta-tag">📐 <b>Aspect Ratio:</b> {meta['aspect_ratio']}</span>
            <span class="meta-tag">💾 <b>File Size:</b> {meta['size_kb']}</span>
            <span class="meta-tag">🎨 <b>Color Mode:</b> {meta['mode']}</span>
            <span class="meta-tag">📊 <b>Shannon Entropy:</b> {meta['entropy']}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border: 2px dashed #cbd5e1; border-radius: 12px; padding: 3rem 1.5rem; text-align: center; color: #94a3b8;">
            <p style="font-size: 2.5rem; margin-bottom: 0.5rem;">🖼️</p>
            <p style="font-weight: 500; color: #64748b;">No image loaded yet</p>
            <p style="font-size: 0.85rem;">Uploaded or sample image preview will appear here</p>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# Model Inference Execution
# ==============================================================================
if active_image is not None and analyze_clicked:
    with st.spinner("Executing forensic pipeline: Computing ELA residuals & running deep CNN inference..."):
        try:
            detector = get_detector(selected_model_key)
            result = detector.predict(active_image, custom_threshold=threshold_slider)
            st.session_state["analysis_results"] = result
            st.session_state["analyzed_image"] = active_image
            st.session_state["analyzed_filename"] = active_filename
        except Exception as e:
            st.error(f"Inference Failure: {str(e)}")
            st.session_state["analysis_results"] = None


# ==============================================================================
# Results Presentation Section
# ==============================================================================
if st.session_state.get("analysis_results") is not None:
    res = st.session_state["analysis_results"]

    st.markdown("---")
    st.markdown("## 3. Forensic Analysis Results")

    # Dynamic Classification Banner
    if res["is_tampered"]:
        banner_class = "result-tampered"
        status_icon = "⚠️"
        headline = "TAMPERED / FORGED"
        desc = "The model detected high-frequency forensic anomalies and compression rate inconsistencies characteristic of image manipulation."
    else:
        banner_class = "result-authentic"
        status_icon = "✅"
        headline = "ORIGINAL / AUTHENTIC"
        desc = "The model detected consistent compression residuals and uniform structural textures characteristic of an untouched camera capture."

    st.markdown(f"""
    <div class="result-banner {banner_class}">
        <div class="result-title">{status_icon} Prediction: {headline}</div>
        <div class="result-subtitle">{desc} &bull; <i>Decision Threshold: &tau; = {res.get('threshold_used', 0.5):.2f}</i></div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Metric Columns
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

    with col_metric1:
        st.metric(
            label="Prediction Confidence",
            value=f"{res['confidence_percent']:.1f}%",
            delta="Predicted Class Score"
        )
    with col_metric2:
        st.metric(
            label="Tampered Probability",
            value=f"{res['prob_tampered_percent']:.1f}%",
            delta=f"Logit Softmax"
        )
    with col_metric3:
        st.metric(
            label="Authentic Probability",
            value=f"{res['prob_original_percent']:.1f}%",
            delta=f"Logit Softmax"
        )
    with col_metric4:
        st.metric(
            label="Inference Latency",
            value=f"{res['latency_ms']:.1f} ms",
            delta=f"Device: {res['device']}"
        )

    # Probability Bars
    st.markdown("#### Probability Distribution:")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.caption(f"Authentic (Original) Probability: {res['prob_original_percent']:.1f}%")
        st.progress(float(res['prob_original']))
    with col_p2:
        st.caption(f"Manipulated (Tampered) Probability: {res['prob_tampered_percent']:.1f}%")
        st.progress(float(res['prob_tampered']))

    # Forensic Visualization Comparison (RGB vs ELA)
    st.markdown("---")
    st.markdown("### 4. Side-by-Side Forensic Visualization")
    st.write(
        "Compare the uploaded image against its **Error Level Analysis (ELA)** map. "
        "ELA highlights compression rate differentials; spliced objects saved at different compression levels "
        "will exhibit pronounced brightness or pattern shifts relative to the background."
    )

    col_vis1, col_vis2 = st.columns(2, gap="medium")
    with col_vis1:
        st.image(
            st.session_state["analyzed_image"],
            caption=f"Original Upload: {st.session_state.get('analyzed_filename', '')}",
            use_container_width=True
        )
    with col_vis2:
        st.image(
            res["ela_image"],
            caption="Error Level Analysis (ELA) Residual Map (Quality=90, Scale=12x)",
            use_container_width=True
        )

    # Academic Forensic Notice
    st.markdown("""
    <div class="forensic-disclaimer">
        <b>⚖️ Forensic Science Disclaimer & Academic Guidance:</b><br>
        This prediction is generated by a custom residual convolutional neural network trained from scratch on the CG-1050 dataset. 
        In digital image forensics, no single deep learning model or standalone ELA map provides absolute proof of manipulation. 
        Re-saving an image in JPEG format multiple times homogenizes compression artifacts, while high-texture surfaces naturally produce elevated ELA residuals. 
        In professional forensics, deep learning estimates must always be combined with Photo Response Non-Uniformity (PRNU) sensor noise verification, JPEG ghost analysis, and EXIF metadata examination.
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# Footer
# ==============================================================================
st.markdown("---")
st.caption("CG-1050 Image Forgery Detection Study | Built with Streamlit, PyTorch, and OpenCV | Academic Capstone Interface")
