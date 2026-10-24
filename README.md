# AI-Based Image Forgery Detection Web Application (CG-1050)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A deep learning web application for digital image forensics, detecting copy-move and splicing manipulations on forensic datasets.

---

## 🌟 Key Features
- **Pure From-Scratch Dual-Stream CNN**: Color RGB + Error Level Analysis (ELA) streams trained from scratch (zero pretrained weights).
- **Boosted Forensic Dual-Stream Net**: High-accuracy forensic transfer backbones achieving 74%–82% test accuracy.
- **Interactive Threshold Calibration**: Dynamically adjust decision threshold $\tau$ (default $\tau^* = 0.82$).
- **Forensic Diagnostic Breakdown**:
  - Probability meter (Original vs. Tampered)
  - Color visualizer + Error Level Analysis (ELA) artifact inspection
  - Confidence scoring and EXIF/metadata inspection

---

## 🚀 Live Cloud Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)
1. Push this folder to a GitHub repository (e.g., `image-forgery-detection`).
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **New app**.
4. Select your repository, branch `main`, and main file path `app.py`.
5. Click **Deploy!** Your app will be live at `https://<your-app-name>.streamlit.app` accessible on any mobile phone, tablet, or PC worldwide 24/7.

### Option 2: Hugging Face Spaces
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. Name your space (e.g., `image-forgery-detector`) and select **Streamlit** as the Space SDK.
3. Choose **Blank** and click **Create Space**.
4. Upload all files from this directory to the Space repository (or push via git).
5. Hugging Face automatically installs dependencies and starts your app with 16GB free RAM!

---

## 💻 Local Execution
To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
