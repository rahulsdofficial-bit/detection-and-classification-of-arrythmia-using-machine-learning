# Dataset Information: MIT-BIH Arrhythmia Database & Supplementary Datasets

## 1. Primary Dataset: MIT-BIH Arrhythmia Database
- **Official Source**: PhysioNet / Massachusetts Institute of Technology & Beth Israel Hospital
- **URL**: [https://physionet.org/content/mitdb/1.0.0/](https://physionet.org/content/mitdb/1.0.0/)
- **Version**: 1.0.0
- **License**: Openly available under the Open Data Commons Public Domain Dedication and License (PDDL) v1.0.
- **Recordings**: 48 half-hour two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979.
- **Sampling Rate**: 360 Hz (360 samples per second per channel).
- **Resolution**: 11-bit resolution over a 10 mV range.
- **Channels**: Channel 1 is typically modified limb lead II (MLII); Channel 2 is typically V1, V2, V4, or V5.
- **Annotations**: Each beat independently verified and labeled by at least two cardiologists.

### File Format Requirements
Each record in `data/mitbih/` consists of:
1. `.dat`: Binary ECG signal recording samples.
2. `.hea`: ASCII header file detailing channel names, sampling rate, units, and calibration constants.
3. `.atr`: Binary reference annotation file marking R-peak locations and beat classifications.

### AAMI Heartbeat Standard Mapping
To adhere to the Association for the Advancement of Medical Instrumentation (AAMI EC57) guidelines and the review paper specifications:
- **N (Normal / Non-ectopic)**: Normal sinus beat (`N`), Left bundle branch block (`L`), Right bundle branch block (`R`), Atrial escape (`e`), Nodal escape (`j`).
- **S (Supraventricular Ectopic Beat - SVEB)**: Atrial premature beat (`A`), Aberrated atrial premature beat (`a`), Nodal premature beat (`J`), Supraventricular premature beat (`S`).
- **V (Ventricular Ectopic Beat - VEB)**: Premature ventricular contraction (`V`), Ventricular escape beat (`E`).
- **F (Fusion Beat)**: Fusion of ventricular and normal beat (`F`).
- **Q (Unknown / Unclassifiable)**: Paced beat (`/`), Fusion of paced and normal (`f`), Unclassifiable beat (`Q`).

### Inter-Patient Train-Test Split (de Chazal Standard)
To strictly prevent data leakage between training and testing sets:
- **Training Set (DS1)**: 22 records (`101`, `106`, `108`, `109`, `112`, `114`, `115`, `116`, `118`, `119`, `122`, `124`, `201`, `203`, `205`, `207`, `208`, `209`, `215`, `220`, `223`, `230`)
- **Testing Set (DS2)**: 22 records (`100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234`)

---

## 2. Supplementary Datasets Investigation

### PTB-XL: Large Electrocardiography Database
- **Source**: PhysioNet ([https://physionet.org/content/ptb-xl/](https://physionet.org/content/ptb-xl/))
- **Description**: 21,799 clinical 12-lead ECG records of 10-second duration from 18,869 patients, sampled at 500 Hz / 100 Hz.
- **Evaluation**: Primarily formatted for multi-label clinical diagnostic statements (e.g., myocardial infarction, ST-T abnormalities, bundle branch blocks) rather than continuous single-lead beat-by-beat arrhythmia segmentation. Download size is >3 GB. Given the project focus on beat-level arrhythmia classification with WKNN, MIT-BIH provides the direct ground truth for beat morphology.

### INCART: St. Petersburg Institute of Cardiological Technics 12-Lead Arrhythmia Database
- **Source**: PhysioNet ([https://physionet.org/content/incartdb/](https://physionet.org/content/incartdb/))
- **Description**: 75 annotated 12-lead ECG records sampled at 257 Hz.
- **Evaluation**: Compatible format, but MIT-BIH is the universal benchmark that matches the paper's experiments and provides sufficient diversity across all 5 AAMI classes.

---

## 3. Storage and Persistence
- Downloaded records are preserved in `data/mitbih/`.
- Pre-extracted test samples are stored in `data/samples/` as `.csv` and `.dat`/`.hea` for fast UI demonstration.
- Existing files are verified by size and checksum; re-downloading is skipped once present.
