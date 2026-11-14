"""
ecg_processing.py
ECG Signal Processing Pipeline:
- Loading (WFDB, CSV)
- Filtering (Butterworth Bandpass 0.5-40Hz, Notch 50/60Hz, Baseline Removal, Normalization)
- R-Peak Detection (Pan-Tompkins Algorithm & Peak Finding)
- Heartbeat Segmentation (+-0.3s window)
- Feature Extraction (Time-domain, Morphological, Statistical, Wavelet, FFT)
"""

import os
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import pywt
import pandas as pd
import wfdb

# AAMI EC57 Class Mapping
AAMI_MAPPING = {
    # Normal / Non-ectopic (N)
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    # Supraventricular Ectopic Beat (S)
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    # Ventricular Ectopic Beat (V)
    'V': 'V', 'E': 'V',
    # Fusion Beat (F)
    'F': 'F',
    # Unknown / Paced Beat (Q)
    '/': 'Q', 'f': 'Q', 'Q': 'Q'
}

AAMI_NAMES = {
    'N': 'Normal Beat',
    'S': 'Supraventricular Ectopic Beat',
    'V': 'Ventricular Ectopic Beat',
    'F': 'Fusion Beat',
    'Q': 'Unknown / Paced Beat'
}

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=360.0, order=3):
    """Zero-phase Butterworth bandpass filter to eliminate baseline drift and high-freq noise."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def notch_filter(data, cutoff=50.0, fs=360.0, q=30.0):
    """IIR notch filter to remove powerline interference (50Hz or 60Hz)."""
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return data
    freq = cutoff / nyq
    b, a = signal.iirnotch(freq, q)
    y = signal.filtfilt(b, a, data)
    return y

def normalize_signal(data):
    """Standardizes ECG signal amplitude using Z-score or robust min-max."""
    std = np.std(data)
    if std < 1e-8:
        return data - np.mean(data)
    return (data - np.mean(data)) / std

def preprocess_ecg(raw_signal, fs=360.0):
    """
    Complete preprocessing pipeline:
    1. Bandpass filter (0.5 - 40 Hz)
    2. 50 Hz notch filter
    3. 60 Hz notch filter
    4. Amplitude normalization
    """
    sig = np.asarray(raw_signal, dtype=np.float64)
    # Bandpass
    filtered = butter_bandpass_filter(sig, lowcut=0.5, highcut=40.0, fs=fs, order=3)
    # Notch filters for 50Hz and 60Hz power lines
    filtered = notch_filter(filtered, cutoff=50.0, fs=fs, q=30.0)
    filtered = notch_filter(filtered, cutoff=60.0, fs=fs, q=30.0)
    # Normalize
    normalized = normalize_signal(filtered)
    return normalized

def pan_tompkins_detector(ecg_signal, fs=360.0):
    """
    Pan-Tompkins algorithm for real-time QRS / R-peak detection.
    Steps:
      1. Bandpass (5-15 Hz)
      2. Derivative
      3. Squaring
      4. Moving Window Integration (MWI)
      5. Adaptive Thresholding
      6. Exact R-peak alignment on filtered signal
    """
    sig = np.asarray(ecg_signal, dtype=np.float64)
    n_samples = len(sig)
    if n_samples < int(fs):
        return np.array([], dtype=int)

    # 1. Bandpass filter 5 - 15 Hz
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = min(0.99, 15.0 / nyq)
    b_bp, a_bp = signal.butter(2, [low, high], btype='band')
    bpass = signal.filtfilt(b_bp, a_bp, sig)

    # 2. Derivative filter: y[n] = (1/8T) * (-x[n-2] - 2x[n-1] + 2x[n+1] + x[n+2])
    der_kernel = np.array([-1, -2, 0, 2, 1]) / 8.0
    derived = np.convolve(bpass, der_kernel, mode='same')

    # 3. Squaring
    squared = derived ** 2

    # 4. Moving Window Integration (approx 150ms window)
    window_len = int(0.150 * fs)
    if window_len < 3:
        window_len = 3
    kernel = np.ones(window_len) / window_len
    integrated = np.convolve(squared, kernel, mode='same')

    # 5. Peak finding on MWI signal
    min_distance = int(0.200 * fs)  # 200ms refractory period
    prominence = np.mean(integrated) * 0.5
    peaks, _ = signal.find_peaks(integrated, distance=min_distance, prominence=prominence)

    if len(peaks) == 0:
        # Fallback to direct prominent peaks on filtered signal
        peaks, _ = signal.find_peaks(sig, distance=min_distance, height=np.mean(sig) + 0.5 * np.std(sig))
        return peaks

    # 6. Refine peak positions to match local maximum in original filtered signal
    refined_peaks = []
    search_window = int(0.075 * fs)  # 75 ms search window around MWI peak
    for p in peaks:
        left = max(0, p - search_window)
        right = min(n_samples, p + search_window)
        if right > left:
            local_max = left + np.argmax(sig[left:right])
            # Avoid duplicate peaks within refractory period
            if not refined_peaks or (local_max - refined_peaks[-1]) > min_distance:
                refined_peaks.append(local_max)

    return np.array(refined_peaks, dtype=int)

def segment_heartbeats(ecg_signal, r_peaks, fs=360.0, window_sec=0.3, pad_boundaries=True):
    """
    Extracts fixed window (+- window_sec, default 0.3s) around each R-peak.
    Parameters:
      ecg_signal: 1D numpy array of ECG signal
      r_peaks: 1D array or list of R-peak sample indices
      fs: sampling frequency (default 360 Hz)
      window_sec: half-window width in seconds (default 0.3s)
      pad_boundaries: if True, pads boundary peaks with zeros; if False, excludes boundary peaks
    Returns:
      beats: list of 1D numpy arrays
      valid_indices: list of original R-peak indices for valid beats
    """
    half_win = int(window_sec * fs)
    beats = []
    valid_peaks = []
    n_samples = len(ecg_signal)

    for r in r_peaks:
        left = r - half_win
        right = r + half_win
        if left >= 0 and right < n_samples:
            seg = ecg_signal[left:right]
            beats.append(seg)
            valid_peaks.append(r)
        elif pad_boundaries and (0 <= r < n_samples):
            # Handle edge cases with zero padding
            seg = np.zeros(2 * half_win, dtype=np.float64)
            start_in_sig = max(0, left)
            end_in_sig = min(n_samples, right)
            sig_slice = ecg_signal[start_in_sig:end_in_sig]
            
            pad_left = max(0, -left)
            seg[pad_left : pad_left + len(sig_slice)] = sig_slice
            beats.append(seg)
            valid_peaks.append(r)
        # If pad_boundaries is False, peaks with left < 0 or right >= n_samples are excluded.

    return beats, valid_peaks

def extract_beat_features(beat_segment, r_idx, all_r_peaks, fs=360.0, beat_idx=None):
    """
    Extracts clinical and ML features from an individual heartbeat:
    - Time-domain: pre-RR, post-RR, local-RR, RR ratio, QRS duration
    - Morphological: R peak amplitude, P wave estimate, T wave estimate, peak-to-peak, energy
    - Statistical: mean, std/variance, skewness, kurtosis
    - Frequency & Wavelet: 4-level DWT subband energies (db4), FFT band powers
    
    Parameters:
      beat_segment: 1D array of ECG signal (+-0.3s around R-peak)
      r_idx: Sample index of current R-peak
      all_r_peaks: 1D ordered array of valid beat locations ONLY (excluding non-beat events)
      fs: Sampling frequency in Hz
      beat_idx: Optional integer index of r_idx in all_r_peaks for direct O(1) lookup
    """
    features = []
    
    if beat_idx is not None and 0 <= beat_idx < len(all_r_peaks):
        idx_in_peaks = beat_idx
    else:
        peak_pos = np.where(all_r_peaks == r_idx)[0]
        idx_in_peaks = peak_pos[0] if len(peak_pos) > 0 else -1

    # 1. RR Interval features using strictly the ordered sequence of valid beats
    if idx_in_peaks > 0:
        pre_rr = (all_r_peaks[idx_in_peaks] - all_r_peaks[idx_in_peaks - 1]) / fs
    else:
        pre_rr = 0.8  # default baseline ~75 bpm

    if 0 <= idx_in_peaks < len(all_r_peaks) - 1:
        post_rr = (all_r_peaks[idx_in_peaks + 1] - all_r_peaks[idx_in_peaks]) / fs
    else:
        post_rr = pre_rr

    # Local average RR (5 beats before and after)
    if idx_in_peaks >= 0:
        start_w = max(0, idx_in_peaks - 5)
        end_w = min(len(all_r_peaks), idx_in_peaks + 6)
        if end_w > start_w + 1:
            diffs = np.diff(all_r_peaks[start_w:end_w]) / fs
            local_rr = np.mean(diffs) if len(diffs) > 0 else pre_rr
        else:
            local_rr = pre_rr
    else:
        local_rr = pre_rr

    rr_ratio = pre_rr / (local_rr + 1e-6)

    features.extend([pre_rr, post_rr, local_rr, rr_ratio])

    # 2. Morphological features
    seg = np.asarray(beat_segment, dtype=np.float64)
    seg_len = len(seg)
    mid = seg_len // 2
    r_amp = seg[mid]

    # P-wave region: approx -0.25s to -0.05s before R
    p_start = max(0, mid - int(0.25 * fs))
    p_end = max(0, mid - int(0.05 * fs))
    p_amp = np.max(seg[p_start:p_end]) if p_end > p_start else 0.0

    # T-wave region: approx +0.05s to +0.30s after R
    t_start = min(seg_len, mid + int(0.05 * fs))
    t_end = min(seg_len, mid + int(0.30 * fs))
    t_amp = np.max(seg[t_start:t_end]) if t_end > t_start else 0.0

    peak_to_peak = np.ptp(seg) if len(seg) > 0 else 0.0
    seg_energy = np.sum(seg ** 2) / (seg_len + 1e-6)

    # QRS duration estimate: full width at half maximum (FWHM) around R peak
    half_max = r_amp * 0.5
    left_q = mid
    while left_q > 0 and seg[left_q] > half_max:
        left_q -= 1
    right_s = mid
    while right_s < seg_len - 1 and seg[right_s] > half_max:
        right_s += 1
    qrs_duration = (right_s - left_q) / fs

    features.extend([r_amp, p_amp, t_amp, peak_to_peak, seg_energy, qrs_duration])

    # 3. Statistical features
    feat_mean = np.mean(seg)
    feat_var = np.var(seg)
    feat_skew = float(stats.skew(seg)) if len(seg) > 2 else 0.0
    feat_kurt = float(stats.kurtosis(seg)) if len(seg) > 2 else 0.0

    features.extend([feat_mean, feat_var, feat_skew, feat_kurt])

    # 4. Wavelet features (db4, level 4 decomposition)
    try:
        coeffs = pywt.wavedec(seg, 'db4', level=4)
        for c in coeffs:
            c_energy = np.sum(c ** 2) / (len(c) + 1e-6)
            features.append(c_energy)
    except Exception:
        features.extend([0.0] * 5)

    # 5. FFT Frequency domain features
    try:
        fft_vals = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
        # Low band (0.5 - 5 Hz)
        low_band = np.sum(fft_vals[(freqs >= 0.5) & (freqs < 5.0)])
        # Mid band (5 - 15 Hz)
        mid_band = np.sum(fft_vals[(freqs >= 5.0) & (freqs < 15.0)])
        # High band (15 - 40 Hz)
        high_band = np.sum(fft_vals[(freqs >= 15.0) & (freqs <= 40.0)])
        dominant_freq = freqs[np.argmax(fft_vals[1:]) + 1] if len(fft_vals) > 1 else 0.0

        features.extend([low_band, mid_band, high_band, dominant_freq])
    except Exception:
        features.extend([0.0] * 4)

    return np.array(features, dtype=np.float64)

FEATURE_NAMES = [
    'pre_rr', 'post_rr', 'local_rr', 'rr_ratio',
    'r_amp', 'p_amp', 't_amp', 'peak_to_peak', 'seg_energy', 'qrs_duration',
    'mean', 'variance', 'skewness', 'kurtosis',
    'dwt_cA4_energy', 'dwt_cD4_energy', 'dwt_cD3_energy', 'dwt_cD2_energy', 'dwt_cD1_energy',
    'fft_low_band', 'fft_mid_band', 'fft_high_band', 'fft_dominant_freq'
]

def load_ecg_file(filepath):
    """
    Loads ECG data from:
    1. WFDB record (.hea, .dat, .atr)
    2. CSV file containing ECG signal values
    Returns:
      signal: 1D numpy array
      fs: sampling frequency (float)
      info: dictionary with metadata
    """
    info = {
        'filepath': filepath,
        'filename': os.path.basename(filepath),
        'fs': 360.0,
        'format': 'unknown'
    }

    base, ext = os.path.splitext(filepath)
    ext = ext.lower()

    if ext in ['.dat', '.hea', '.atr'] or os.path.exists(base + '.hea'):
        # WFDB record
        record_dir = os.path.dirname(filepath) or '.'
        record_name = os.path.basename(base)
        rec_path = os.path.join(record_dir, record_name)
        record = wfdb.rdrecord(rec_path)
        sig = record.p_signal[:, 0]  # First lead (usually MLII)
        info['fs'] = float(record.fs)
        info['format'] = 'WFDB'
        info['sig_name'] = record.sig_name[0] if record.sig_name else 'Lead I'
        info['units'] = record.units[0] if record.units else 'mV'
        return sig, info['fs'], info

    elif ext == '.csv':
        df = pd.read_csv(filepath)
        info['format'] = 'CSV'
        # Detect voltage column
        possible_cols = ['ecg', 'voltage', 'signal', 'lead_ii', 'mlii', 'val', 'data']
        col_name = None
        for col in df.columns:
            if any(p in col.lower() for p in possible_cols):
                col_name = col
                break
        if col_name is None:
            # Fallback to the last numerical column or column 1
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            col_name = numeric_cols[-1] if len(numeric_cols) > 0 else df.columns[0]

        sig = df[col_name].dropna().to_numpy(dtype=np.float64)

        # Detect sample rate from time column if present
        time_cols = [c for c in df.columns if 'time' in c.lower()]
        if time_cols:
            t_diff = np.diff(df[time_cols[0]].dropna().values)
            t_diff = t_diff[t_diff > 0]
            if len(t_diff) > 0 and np.median(t_diff) > 0:
                calc_fs = 1.0 / np.median(t_diff)
                if 50 <= calc_fs <= 2000:
                    info['fs'] = float(round(calc_fs))

        return sig, info['fs'], info

    else:
        raise ValueError(f"Unsupported file format: {ext}. Please provide .csv or WFDB record files.")
