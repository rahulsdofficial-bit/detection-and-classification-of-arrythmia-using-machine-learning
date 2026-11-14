# Detection and Classification of Arrhythmia Using Machine Learning

An autonomous, full-stack clinical decision support web application and machine learning framework for detecting and classifying cardiac arrhythmias from Electrocardiogram (ECG) signals using the proposed **Weighted K-Nearest Neighbors (WKNN)** algorithm.

Developed based on the research specifications of SJBIT Department of Information Science & Engineering / Computer Science & Engineering.

---

> [!IMPORTANT]
> **Academic Research Prototype Disclaimer:** This software is an academic research decision-support prototype and is not intended for clinical diagnosis or treatment. All findings require independent review by qualified healthcare professionals.

---

## 📌 Project Overview

Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide. ECG analysis is the primary diagnostic tool for identifying arrhythmias; however, manual interpretation is time-consuming and subject to human fatigue.

This application provides an end-to-end automated platform:
1. **ECG Signal Preprocessing**: Butterworth bandpass filtering (0.5–40 Hz), 50/60 Hz IIR notch filtering, baseline wander removal, and amplitude normalization.
2. **QRS / R-Peak Detection**: Real-time **Pan-Tompkins algorithm** featuring derivative filtering, squaring, and adaptive moving window integration.
3. **Heartbeat Segmentation**: Cardiac cycle isolation around R-peaks (&plusmn;0.3 seconds).
4. **Multi-Domain Feature Extraction**: 23 clinical features including RR intervals, QRS width, wavelet subband energies (PyWavelets DWT `db4`), FFT spectral band powers, and higher-order statistics (skewness, kurtosis).
5. **Machine Learning Classification**: Primary proposed **Weighted K-Nearest Neighbors (WKNN)** classifier evaluated alongside baseline models (Standard KNN, SVM, Logistic Regression) under the strict **de Chazal et al. inter-patient separation protocol** (DS1 for training, DS2 for testing).
6. **AAMI EC57 Standardization**: Heartbeat categorization into **N** (Normal), **S** (Supraventricular Ectopic), **V** (Ventricular Ectopic), **F** (Fusion), and **Q** (Unknown / Paced).
7. **Interactive Web Application**: Responsive clinician portal built with Flask, SQLite, Bootstrap 5, and Plotly.js.
8. **Automated Clinical Reporting**: PDF medical diagnostic reports generated via ReportLab with ECG rhythm strips and beat-by-beat CSV exports.

---

## 📊 Rigorous Experimental Evaluation (de Chazal Inter-Patient Protocol)

To prevent data leakage, 44 patient records from the MIT-BIH Arrhythmia Database were strictly divided into two non-overlapping cohorts (de Chazal et al.):
- **Training Cohort (DS1)**: 22 records (`101`, `106`, `108`, `109`, `112`, `114`, `115`, `116`, `118`, `119`, `122`, `124`, `201`, `203`, `205`, `207`, `208`, `209`, `215`, `220`, `223`, `230`) — 51,006 raw valid beats (balanced to 6,366 beats by capping N at 2,500 and retaining all minority S, V, F, Q beats).
- **Testing Cohort (DS2)**: 22 records (`100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234`) — **49,697 complete, un-balanced test beats** evaluated without artificial downsampling.
- **Excluded Records**: Paced records `102`, `104`, `107`, `217` per AAMI standard. Non-beat annotations (`+`, `~`, `|`, `x`) and boundary beats (< 0.3s) were excluded.

### 1. Annotation-Based Evaluation (Complete Un-balanced DS2 Test Population: 49,697 Beats)

| Algorithm | Overall Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Macro F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Weighted K-NN (WKNN)** | **82.17%** | **90.82%** | **82.17%** | **85.68%** | **39.31%** | **Proposed Primary Model** |
| Standard K-NN | 83.00% | 91.00% | 83.00% | 86.28% | 40.13% | Baseline |
| Support Vector Machine (SVM) | 88.93% | 92.24% | 88.93% | 90.16% | 43.22% | Baseline |
| Logistic Regression | 86.17% | 91.83% | 86.17% | 88.55% | 40.75% | Baseline |

#### Proposed WKNN Per-Class Clinical Sensitivity on Unseen DS2 Patients:
- **Class N (Normal)**: **83.9%** sensitivity (37,126 / 44,245 beats)
- **Class S (Supraventricular Ectopic)**: **41.5%** sensitivity (763 / 1,837 beats)
- **Class V (Ventricular Ectopic / PVCs)**: **87.7%** sensitivity (2,823 / 3,220 beats)
- **Class F (Fusion)**: **33.0%** sensitivity (128 / 388 beats)
- **Class Q (Unknown / Paced)**: **0.0%** sensitivity (0 / 7 beats)

### 2. End-to-End Pipeline Evaluation (Continuous Signals on DS2: Pan-Tompkins + WKNN)
- **Acceptance Tolerance Window**: &plusmn;150 ms (&plusmn;54 samples at 360 Hz)
- **Reference Beats**: 49,712
- **Detected Peaks**: 50,088
- **True Positives (TP)**: 49,643
- **False Positives (FP)**: 445
- **False Negatives (FN)**: 69
- **Pan-Tompkins QRS Detection Sensitivity ($Se$)**: **99.86%**
- **Pan-Tompkins Positive Predictivity ($+P$)**: **99.11%**
- **End-to-End Classification Accuracy on Matched Beats**: **83.07%** (Weighted F1: **86.49%**)

---

## 🔍 Verified Benchmark Demo Samples

Each test sample in `data/samples/` was independently verified against MIT-BIH reference annotations:
- **Record 100** (`sample_100_Normal_Sinus_Rhythm_with_APC.csv`): 12 Normal beats, 1 Atrial Premature Contraction (APC/SVEB).
- **Record 119** (`sample_119_Ventricular_Bigeminy_PVCs.csv`): 8 Normal beats, 2 Premature Ventricular Contractions (PVCs/VEBs).
- **Record 200** (`sample_200_Ventricular_Tachycardia_PVCs.csv`): 7 Normal beats alternating with 8 PVCs (**Ventricular Bigeminy**).
- **Record 208** (`sample_208_Ventricular_Couplets_and_Runs.csv`): 5 Normal beats, 9 PVCs featuring **Ventricular Couplets (PVC pairs)**, and 2 Fusion beats.

---

## 🚀 Quick Start Guide

### Option 1: 1-Click Windows Batch Scripts
1. Double-click `setup.bat` to verify dependencies and model artifacts.
2. Double-click `run.bat` to launch the server.

### Option 2: Command Line Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the application:
   ```bash
   python app.py
   ```
3. Open your browser at:
   ```
   http://127.0.0.1:5000
   ```

### Default Clinician Login
- **Username**: `doctor`
- **Password**: `doctor123`

---

## 🛡️ Security Features
- Secure CSRF and session secrets generated via `secrets.token_hex(32)`.
- Password hashing with Werkzeug PBKDF2/SHA-256.
- Strict file extension validation (`.csv`, `.dat`, `.hea`) and filename sanitization via `secure_filename`.
- Secrets in `.env` and `.ethereal_account.json` are excluded from version control via `.gitignore`.
- Fallback mechanisms stop execution cleanly if models are absent — fake predictions are never generated.
