# MRI + Tabular Cognitive Classifier - Frontend Integration Guide

## Overview

This package provides a **3-stage hierarchical classifier** for differential diagnosis of cognitive impairment using:
- **187 tabular clinical features** (demographics, cognitive tests, medications, etc.)
- **1 structural brain MRI** (T1-weighted, NIfTI format)

The classifier outputs:
1. **Prediction JSON** with hierarchical diagnoses and probabilities
2. **GradCAM++ heatmap** as NIfTI for visual overlay on the brain MRI
3. **Skull-stripped brain** as NIfTI for overlay base
4. **SHAP explanations** for the 187 tabular features + MRI importance

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
pip install -e nmed2024/
```

### GPU (Recommended)

The model runs on CUDA GPU. CPU inference is supported but significantly slower.

```python
# Auto-detect GPU
api = MRICognitiveClassifierAPI()

# Force CPU
api = MRICognitiveClassifierAPI(device='cpu')
```

---

## Quick Start

```python
from api import MRICognitiveClassifierAPI

# Initialize (loads all 3 stage models)
api = MRICognitiveClassifierAPI()

# Fast prediction with heatmap (no SHAP)
result = api.predict(
    mri_path='patient_scan.nii.gz',
    features={'his_NACCAGE': 72, 'his_SEX': 1, 'bat_NACCMMSE': 22, ...},
    output_dir='output/patient_001/'
)

print(result['final_diagnosis'])  # e.g., 'AD', 'NC', 'MCI'
print(result['stage1']['de_probability'])  # Dementia probability
```

---

## Input Format

### MRI Scan

- **Format**: NIfTI (`.nii` or `.nii.gz`)
- **Type**: T1-weighted structural MRI
- **Processing**: The pipeline automatically:
  1. Skull-strips the scan (removes non-brain tissue)
  2. Resizes to 128x128x128
  3. Normalizes to [0, 1]

### Tabular Features (JSON)

187 clinical features as a JSON object. Set missing features to `null` or `-4`.

```json
{
    "his_NACCAGE": 72,
    "his_SEX": 1,
    "his_EDUC": 16,
    "bat_NACCMMSE": 22,
    "bat_MOCATOTS": 18,
    "ph_NACCBMI": 25.5,
    "med_NACCAMD": null,
    "faq_BILLS": null,
    ...
}
```

**Feature categories** (prefix mapping):
| Prefix | Category | Count |
|--------|----------|-------|
| `his_` | Demographics/History | ~30 |
| `med_` | Medications | ~20 |
| `ph_` | Physical Measurements | ~10 |
| `bat_` | Cognitive Tests (MMSE, MoCA) | ~50 |
| `exam_` | Neurological Exam | ~30 |
| `cvd_` | Cardiovascular | ~10 |
| `updrs_` | Parkinson's Rating | ~10 |
| `npiq_` | Neuropsychiatric | ~15 |
| `gds_` | Depression Scale | ~5 |
| `faq_` | Functional Assessment | ~10 |

See `docs/feature_descriptions.csv` for the complete list with descriptions.

**Missing data handling**: The model uses transformer attention masking to handle missing features. You can omit features or set them to `null` / `-4`. The model will still produce predictions using available features.

---

## Output Format

### Prediction JSON

```json
{
    "timestamp": "2026-03-05T14:30:00",
    "mri_path": "patient_scan.nii.gz",
    "stage1": {
        "prediction": "DE",
        "de_probability": 0.82,
        "confidence": 0.82
    },
    "stage2": null,
    "stage3": {
        "prediction": "AD",
        "probabilities": {
            "AD": 0.65, "LBD": 0.12, "VD": 0.08,
            "FTD": 0.06, "PRD": 0.02, "NPH": 0.03,
            "SEF": 0.01, "PSY": 0.02, "ODE": 0.01
        },
        "confidence": 0.65
    },
    "final_diagnosis": "AD",
    "heatmaps": {
        "heatmap_nifti": "output/heatmap.nii.gz",
        "brain_nifti": "output/brain.nii.gz",
        "overlay_nifti": "output/overlay.nii.gz",
        "stage": "stage3",
        "target_class": 0,
        "method": "gradcam_pp"
    }
}
```

### Hierarchical Logic

```
Patient Input (MRI + 187 features)
         |
    [Stage 1: Binary]
    DE vs Non-DE
   /              \
Dementia (DE>=0.5)  Non-Dementia (DE<0.5)
    |                      |
[Stage 3: 9-class]   [Stage 2: 3-class]
 AD, LBD, VD, FTD,    NC, MCI, iMCI
 PRD, NPH, SEF,
 PSY, ODE
```

- If **Stage 1** predicts Dementia → `stage2` is `null`, `stage3` has subtype probabilities
- If **Stage 1** predicts Non-Dementia → `stage3` is `null`, `stage2` has class probabilities
- `final_diagnosis` is always set to the terminal prediction

---

## Heatmap Overlay

The heatmap output files share the **same affine matrix and voxel space** as `brain.nii.gz`, so they can be directly overlaid.

### Output Files

| File | Description | Data Type |
|------|-------------|-----------|
| `heatmap.nii.gz` | Raw attribution values [0, 1] | float32 |
| `brain.nii.gz` | Skull-stripped brain | float32 |
| `overlay.nii.gz` | Pre-blended overlay (70% brain + 30% heatmap) | float32 |

### Frontend Overlay

For custom overlay in the frontend:

```python
import nibabel as nib

# Load both files
brain = nib.load('output/brain.nii.gz').get_fdata()
heatmap = nib.load('output/heatmap.nii.gz').get_fdata()

# Custom overlay with adjustable opacity
opacity = 0.4  # User-adjustable slider
overlay = (1 - opacity) * brain_normalized + opacity * heatmap

# Use any colormap: hot, inferno, jet, etc.
# heatmap values are [0, 1] -- map to your preferred colormap
```

### Colormap Recommendation

We recommend a **red-yellow-white** colormap (matches SPM/FSL neuroimaging convention):
- `0.0` = transparent (no attribution)
- `0.3` = dark red (low attribution)
- `0.6` = red-orange (medium attribution)
- `0.8` = yellow (high attribution)
- `1.0` = white (peak attribution)

---

## SHAP Explanations

For interpretable predictions, use `predict_with_explanations()`:

```python
result = api.predict_with_explanations(
    mri_path='scan.nii.gz',
    features=patient_data,
    output_dir='output/',
    n_top_features=20
)

# Top contributing features
for feat in result['feature_importance']['top_features']:
    print(f"{feat['feature']}: {feat['shap_value']:+.4f} ({feat['direction']})")

# MRI importance score
print(f"MRI importance: {result['feature_importance']['mri_importance']:.4f}")
```

### SHAP Output Format

```json
{
    "feature_importance": {
        "tabular_shap": {
            "feature_names": ["his_NACCAGE", "bat_NACCMMSE", ...],
            "shap_values": [0.003, -0.042, ...],
            "base_value": 0.15,
            "target": "DE"
        },
        "mri_importance": 0.15,
        "top_features": [
            {
                "feature": "bat_NACCMMSE",
                "description": "Cognitive Test: NACCMMSE",
                "shap_value": -0.042,
                "direction": "decreases",
                "importance": 0.042
            }
        ]
    }
}
```

**Note**: SHAP computation takes ~3-5 minutes per patient. Use `predict()` for fast inference without SHAP.

---

## Batch Processing

```python
patients = [
    {'mri_path': 'scan1.nii.gz', 'features': {...}},
    {'mri_path': 'scan2.nii.gz', 'features': {...}},
]

results = api.predict_batch(patients, output_dir='batch_output/')
```

Each patient gets a subdirectory: `batch_output/patient_0/`, `batch_output/patient_1/`, etc.

---

## Performance

| Operation | Time | GPU | Notes |
|-----------|------|-----|-------|
| Inference (3 stages) | ~2s | Yes | Fast prediction |
| Heatmap generation | ~5s | Yes | GradCAM++ |
| SHAP explanations | ~3-5min | Yes | KernelExplainer |
| Skull stripping | ~10s | Yes | deepbet |

---

## Error Handling

```python
result = api.predict(mri_path, features, output_dir='output/')

# Check for errors in batch mode
if 'error' in result:
    print(f"Error: {result['error']}")
```

Missing features are handled gracefully -- the model uses attention masking to ignore missing inputs.

---

## API Reference

### `MRICognitiveClassifierAPI`

| Method | Description | Speed |
|--------|-------------|-------|
| `predict(mri_path, features, output_dir)` | Fast inference + optional heatmap | ~7s |
| `predict_with_explanations(mri_path, features, output_dir)` | Full inference + SHAP + heatmap | ~5min |
| `predict_batch(patients, output_dir)` | Batch prediction | ~7s/patient |
| `get_feature_info()` | Get 188 feature names + descriptions | instant |

### Command Line

```bash
# Fast inference with heatmap
python inference_pipeline.py \
    --mri_path scan.nii.gz \
    --features_json features.json \
    --output_dir output/ \
    --generate_heatmap

# With SHAP explanations
python inference_pipeline.py \
    --mri_path scan.nii.gz \
    --features_json features.json \
    --output_dir output/ \
    --with_shap
```
