"""
download_dataset.py
Automated acquisition of the MIT-BIH Arrhythmia Database from PhysioNet.
Validates file presence, supports resume/skip, and creates ready-to-test sample files.
"""

import os
import sys
import time
import urllib.request
import pandas as pd
import wfdb

# Official MIT-BIH records list (48 records)
MITBIH_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# Core subset covering standard de Chazal benchmark and all key arrhythmia types
CORE_RECORDS = [
    # Train set (DS1) representative records:
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119',
    '122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230',
    # Test set (DS2) representative records:
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
]

BASE_URL = "https://physionet.org/files/mitdb/1.0.0"

def is_record_complete(target_dir, record_id):
    for ext in ['.dat', '.hea', '.atr']:
        fpath = os.path.join(target_dir, f"{record_id}{ext}")
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            return False
    return True

def download_file_direct(url, dest_path, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ArrhythmiaML/1.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
                out_file.write(response.read())
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                print(f"  [ERROR] Failed to download {url}: {e}")
                return False

def download_record(record_id, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    if is_record_complete(target_dir, record_id):
        return True, "already exists"

    success = True
    # Try wfdb dl_files first
    try:
        wfdb.dl_files('mitdb', target_dir, [f"{record_id}.dat", f"{record_id}.hea", f"{record_id}.atr"])
        if is_record_complete(target_dir, record_id):
            return True, "downloaded via wfdb"
    except Exception:
        pass

    # Fallback to direct HTTP download from PhysioNet
    for ext in ['.dat', '.hea', '.atr']:
        dest = os.path.join(target_dir, f"{record_id}{ext}")
        if not os.path.exists(dest) or os.path.getsize(dest) == 0:
            url = f"{BASE_URL}/{record_id}{ext}"
            if not download_file_direct(url, dest):
                success = False

    if success and is_record_complete(target_dir, record_id):
        return True, "downloaded via direct HTTP"
    return False, "incomplete"

def download_mitbih(target_dir='data/mitbih', records=None):
    if records is None:
        records = CORE_RECORDS

    print(f"=== Downloading MIT-BIH Arrhythmia Database ({len(records)} records) to {target_dir} ===")
    os.makedirs(target_dir, exist_ok=True)
    
    downloaded = 0
    skipped = 0
    failed = []

    for i, rec in enumerate(records, 1):
        status, msg = download_record(rec, target_dir)
        if status:
            if "already" in msg:
                skipped += 1
            else:
                downloaded += 1
            print(f"[{i}/{len(records)}] Record {rec}: OK ({msg})")
        else:
            failed.append(rec)
            print(f"[{i}/{len(records)}] Record {rec}: FAILED")

    print("\nDownload Summary:")
    print(f"  Total records requested: {len(records)}")
    print(f"  Downloaded: {downloaded}")
    print(f"  Already present: {skipped}")
    print(f"  Failed: {len(failed)}")

    return len(failed) == 0

def create_sample_files(mitbih_dir='data/mitbih', samples_dir='data/samples'):
    """
    Extracts short 10-second segments from key records to data/samples/
    both as CSV and copying WFDB records for easy user testing.
    """
    os.makedirs(samples_dir, exist_ok=True)
    sample_records = [
        ('100', 'Normal_Sinus_Rhythm_with_APC'),
        ('119', 'Ventricular_Bigeminy_PVCs'),
        ('200', 'Ventricular_Tachycardia_PVCs'),
        ('208', 'Ventricular_Couplets_and_Runs'),
    ]

    print("\n=== Generating Preloaded Test Samples in data/samples/ ===")
    for rec_id, desc in sample_records:
        rec_path = os.path.join(mitbih_dir, rec_id)
        if not os.path.exists(f"{rec_path}.hea"):
            continue

        try:
            # Read first 10 seconds (3600 samples at 360 Hz)
            record = wfdb.rdrecord(rec_path, sampto=3600)
            sig = record.p_signal[:, 0]  # Lead MLII
            time_axis = [i / record.fs for i in range(len(sig))]

            csv_name = f"sample_{rec_id}_{desc}.csv"
            csv_path = os.path.join(samples_dir, csv_name)
            df = pd.DataFrame({'time_seconds': time_axis, 'ecg_voltage': sig})
            df.to_csv(csv_path, index=False)

            # Also create a small WFDB record copy in samples
            wfdb.wrsamp(
                record_name=f"sample_{rec_id}",
                fs=record.fs,
                units=record.units[:1],
                sig_name=record.sig_name[:1],
                p_signal=record.p_signal[:3600, :1],
                fmt=['16'],
                write_dir=samples_dir
            )
            print(f"  Created sample: {csv_name} and sample_{rec_id} WFDB files")
        except Exception as e:
            print(f"  Could not create sample for record {rec_id}: {e}")

if __name__ == '__main__':
    target = os.path.join('data', 'mitbih')
    samples = os.path.join('data', 'samples')
    
    # Download core dataset
    download_mitbih(target)
    create_sample_files(target, samples)
    print("\nDataset preparation complete!")
