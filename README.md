# AI-Based Image Forgery Detection & Digital Image Processing Study (CG-1050)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org/)

An academic and practical deep learning repository for **Digital Image Forensics**, featuring:
1. **Interactive Streamlit Web Interface** for real-time image authenticity testing.
2. **Improved & Audited Jupyter Notebooks** implementing Digital Image Processing (DIP) transforms, classical filtering, frequency domain analysis, and from-scratch CNN architectures.
3. **Rigorous Experimental Framework** with 0-scene-leakage evaluation, calibrated decision thresholds, and reproducible scripts.

---

## 📁 Repository Structure

```text
├── app.py                                              # Streamlit web application
├── inference.py                                        # Inference controller & model registry
├── models.py                                           # PyTorch custom CNN architectures
├── preprocessing.py                                    # Image transforms, ELA generator, tensor loaders
├── dip_pipeline.py                                     # Comprehensive Digital Image Processing pipeline
│
├── CG1050_DIP_Study_and_Model_Improvement.ipynb        # Complete DIP study & improved CNN notebook (rendered outputs)
├── CG1050_DIP_Study_and_Model_Improvement_FIXED.ipynb  # Audited zero-leakage, from-scratch notebook
├── CG1050_Custom_CNN.ipynb                             # 105-cell end-to-end Custom CNN notebook (Parts I & II)
├── CG1050_Notebook_Audit_and_Correction_Report.md      # Systematic audit & methodological correction report
│
├── checkpoints/                                        # Trained model checkpoints
│   ├── custom_dual_stream_scratch.pth                  # From-scratch Dual-Stream (RGB + ELA) CNN
│   ├── dual_stream_rgb_ela_best.pth                    # Boosted Forensic Dual-Stream Net (75%–82% accuracy)
│   ├── custom_cnn_official_benchmark_best.pth          # Official CG-1050 benchmark scratch model
│   ├── custom_clahe_128.pth                            # CLAHE luminance stream scratch model
│   └── custom_baseline_128.pth                         # Raw RGB baseline scratch model
│
├── outputs/                                            # Empirical evaluation results & visualizations
│   ├── ablation_study_results.csv                      # Full quantitative metrics table
│   ├── ablation_comparison.png                         # Visual comparison of DIP filter variants
│   ├── final_test_confusion_matrix.png                 # Test partition confusion matrix
│   ├── dip_analysis/                                   # Quantitative metrics (PSNR, SSIM, Entropy)
│   └── dip_visualizations/                             # Multi-spectral DIP filter outputs
│
├── reports/                                            # Comprehensive project documentation
│   ├── CG1050_Image_Forgery_Detection_Project_Report.docx
│   └── CG1050_Xception_Model_Writeup_and_Layers_final.pdf
│
├── .streamlit/config.toml                              # Streamlit cloud server & theme settings
└── requirements.txt                                    # Cloud-ready PyTorch & OpenCV dependencies
```

---

## 🔬 Scientific Methodology & Improvements

### 1. Strict 0-Scene Leakage Partitioning
The CG-1050 dataset comprises 100 base scenes (`Im1` through `Im100`). Images are partitioned **strictly by base scene identifier**:
- **Training Set:** 1,436 images across 69 base scenes (718 Original, 718 Tampered)
- **Validation Set:** 336 images across 16 base scenes (168 Original, 168 Tampered)
- **Test Set:** 316 images across 15 base scenes (158 Original, 158 Tampered)
- **Guarantee:** $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$ (Zero background scene memorization).

### 2. Digital Image Processing (DIP) Pipeline
Implemented across 7 fundamental transform families in [`dip_pipeline.py`](dip_pipeline.py):
- **Intensity Transformations:** Logarithmic transform, power-law (Gamma $\gamma = 0.5, 2.0$), contrast stretching.
- **Histogram Processing:** Global HE, Perceptual LAB-CLAHE, HSV Equalization.
- **Spatial Noise Filtering:** Gaussian ($\sigma=1.0$), Median ($3 \times 3$), Bilateral edge-preserving filtering.
- **Sharpening:** Laplacian second-derivative operator, High-Boost filtering ($A=1.5$), Unsharp Masking.
- **Frequency Domain:** 2D Fast Fourier Transform (FFT), Ideal / Butterworth / Gaussian Low-Pass & High-Pass filters.
- **Edge Detection:** Sobel horizontal/vertical, Prewitt gradient, Canny multi-stage hysteresis.
- **Morphological Processing:** Dilation, erosion, opening, closing, top-hat/black-hat texture extraction.
- **Forensic Error Level Analysis (ELA):** JPEG compression residual extraction highlighting non-uniform quantization artifacts.

---

## 📊 Empirical Ablation Results

| Architecture / Technique | Input Representation | Test Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Custom Dual-Stream (Calibrated $\tau^*=0.82$)** | **RGB + ELA Residuals** | **64.87%** | **64.87%** | **59.31%** | **86.71%** | **0.7044** | **0.7012** |
| Custom Residual CNN + CLAHE | LAB-CLAHE ($128 \times 128$) | 60.76% | 60.76% | 58.60% | 68.99% | 0.6337 | 0.6461 |
| Official Benchmark Custom CNN | Raw RGB ($128 \times 128$) | 59.71% | 59.71% | 58.74% | 62.42% | 0.6053 | 0.6302 |
| Custom Residual Baseline | Raw RGB ($128 \times 128$) | 59.18% | 59.18% | 57.07% | 71.52% | 0.6348 | 0.6460 |
| Boosted Dual-Stream Net (Transfer) | RGB + ELA ($224 \times 224$) | **74%–82%** | **78.50%** | **78.20%** | **76.00%** | **0.7708** | **0.8103** |

---

## 🚀 Running the Streamlit Web App

### Run Locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Deploy to Streamlit Community Cloud (24/7 Free Hosting):
1. Connect this GitHub repository to [share.streamlit.io](https://share.streamlit.io).
2. Select **Branch:** `main` and **Main file path:** `app.py`.
3. Click **Deploy!**

---

## 📓 Running the Jupyter Notebooks
Launch either notebook in JupyterLab, VS Code, or Google Colab:
```bash
jupyter notebook CG1050_DIP_Study_and_Model_Improvement.ipynb
```
or for the from-scratch audited version:
```bash
jupyter notebook CG1050_DIP_Study_and_Model_Improvement_FIXED.ipynb
```
