# ECG Arrhythmia Monitor – README

> **DISCLAIMER:** This is a **research prototype** developed as a course project.  
> It is **NOT** a certified medical device, and **must not** be used for clinical
> diagnosis, patient-care decisions, or any real-world medical application.

---

## Overview

A self-personalizing, on-device ECG arrhythmia monitor implemented in MATLAB.
The pipeline covers:

| Phase | Script | Description |
|-------|--------|-------------|
| 1 | `preprocess.m` | Data loading, band-pass filtering, Pan-Tompkins R-peak detection, beat segmentation & RR features |
| 2 | `train_cnn.m`  | Tiny 1-D CNN (<20 k params) with embedding layer; handles class imbalance |
| 3 | `quantize_model.m` | INT8 post-training quantization; size, MACs & latency report |
| 4 | `personalize.m`    | Patient-specific Mahalanobis classifier + optional last-layer fine-tuning |
| 5 | `stream_demo.m`    | Real-time streaming simulation with coloured beat labels and alarm overlay |
| — | `main.m`           | Master driver – runs all phases sequentially |

---

## Required MATLAB Toolboxes

| Toolbox | Purpose | Required? |
|---------|---------|-----------|
| **Signal Processing Toolbox** | `butter`, `filtfilt`, `resample` | **Yes** |
| **Deep Learning Toolbox** | CNN training, `activations`, `classify` | **Yes** |
| **Statistics and Machine Learning Toolbox** | `confusionmat`, `perfcurve`, `cov` | **Yes** |
| Deep Learning Toolbox **Model Quantization Library** (add-on) | `dlquantizer`, `calibrate`, `validate` | Optional |
| **MATLAB Coder** | C-code generation stub in Phase 3 | Optional |
| **Embedded Coder** | ARM-compute target config | Optional |
| **WFDB Toolbox for MATLAB** | Live PhysioNet download | Optional |

> Install the optional add-ons from **Home → Add-Ons → Get Add-Ons** or MATLAB
> Add-On Explorer.

---

## Data Setup

### Option A – WFDB Toolbox (live download)
1. Install the [WFDB Toolbox for MATLAB](https://physionet.org/content/wfdb-matlab/)
2. Run `main(true)` – records are streamed from PhysioNet on first access.

### Option B – Pre-exported `.mat` files (recommended for offline use)
Place files in `./data/<recID>.mat` with these workspace variables:

```matlab
val        % double [N_samples × N_leads]  – ECG signal
Fs         % double scalar                 – sampling rate (Hz)
ann_sample % double [N_ann × 1]            – beat annotation sample indices
ann_type   % cell   {N_ann × 1}            – beat type strings e.g. {'N','V',…}
```

Export from Python / WFDB:
```python
import wfdb, scipy.io, numpy as np
rec = wfdb.rdrecord('mitdb/100')
ann = wfdb.rdann('mitdb/100', 'atr')
scipy.io.savemat('data/100.mat', {
    'val': rec.p_signal, 'Fs': rec.fs,
    'ann_sample': ann.sample, 'ann_type': np.array(ann.symbol, dtype=object)
})
```

### Option C – Synthetic demo (no data required)
Run `main()` with no arguments. Each missing record falls back to a synthetic
ECG with random beat labels so every phase can be exercised end-to-end.

---

## Inter-Patient Split (de Chazal 2004)

| Split | Records |
|-------|---------|
| **DS1 (train)** | 101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230 |
| **DS2 (test)** | 100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234 |
| Excluded (paced) | 102 104 107 217 |

## AAMI Class Mapping

| AAMI | MIT-BIH beat types |
|------|--------------------|
| **N** | N L R e j |
| **S** | A a J S |
| **V** | V E |
| **F** | F |
| **Q** | / f Q |

---

## Quick Start

```matlab
% Full pipeline (synthetic data – no toolboxes beyond Signal + DL + Stats)
main()

% Full pipeline with WFDB live download
main(true)

% Load cached model, run streaming demo on DS2 record #2
main(false, 2)

% Run individual phases
[DS1, DS2] = preprocess(false);
[net, info, trainData, testData] = train_cnn(DS1, DS2);
quantize_model(net, testData);
personalize(net, DS2);
stream_demo(net, DS2, 1);
```

---

## Outputs

```
figures/
  phase2_confusion.png      – DS2 confusion matrix
  phase2_roc.png            – DS2 ROC curves (one-vs-rest)
  phase3_quantization.png   – FP32 vs INT8 accuracy bar chart
  phase4_personalization.png – Before/after personalisation
  phase5_latency.png        – Per-beat inference latency histogram
  phase5_ecg_snapshot.png   – Annotated streaming ECG

results/
  quantization_results.csv  – Size, accuracy, MACs, RAM, latency
  personalization_results.csv – Per-patient sensitivity/PPV/FAR

data/
  preprocessed.mat          – DS1/DS2 beat arrays (cached after Phase 1)
  cnn_model.mat             – Trained network (cached after Phase 2)
  train_test.mat            – Train/test arrays (cached after Phase 2)
  quantized_model.mat       – Quantized dlquantizer object (Phase 3, optional)
```

---

## Architecture Details

### CNN (Phase 2)

```
Input [200×1×1]
  Conv1(16 filters, k=7) → BN → ReLU → MaxPool(2)   → [100×1×16]
  Conv2(32 filters, k=5) → BN → ReLU → MaxPool(2)   → [50×1×32]
  Conv3(32 filters, k=5) → BN → ReLU → MaxPool(2)   → [25×1×32]
  GlobalAveragePooling                               → [32]
  FC-64 (embedding) → ReLU → Dropout(0.3)            → [64]
  FC-5  → Softmax                                    → [5] AAMI classes
Total parameters ≈ 10,500  (< 20 k target ✓)
```

### Personalisation (Phase 4)

```
Calibration window (first 5 min of patient record)
  → Extract 64-D CNN embeddings of Normal beats
  → Fit Gaussian: μ_N, Σ_N  (regularised)

Inference
  ‣ Mahalanobis distance d = √((e − μ_N)ᵀ Σ_N⁻¹ (e − μ_N))
  ‣ d ≤ threshold → label N  (personalised)
  ‣ d  > threshold → use base CNN prediction

Optional fine-tuning
  → SGD (3 epochs, lr=1e-4) on last FC layer
     using confirmed-Normal beats from calibration
```

---

## Citation / References

- Pan & Tompkins, "A Real-Time QRS Detection Algorithm," *IEEE TBME*, 1985.
- de Chazal et al., "Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features," *IEEE TBME*, 2004.
- MIT-BIH Arrhythmia Database, PhysioNet. https://physionet.org/content/mitdb/

---

*Research prototype. Not a medical device.*

