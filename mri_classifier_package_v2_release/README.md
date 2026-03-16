# MRI + Tabular Cognitive Classifier v2.0

Hierarchical classification framework for **differential diagnosis of cognitive impairment** using structural brain MRI and 187 tabular clinical features.

## What's New in v2.0

| Feature | v1.0 (Tabular Only) | v2.0 (MRI + Tabular) |
|---------|---------------------|----------------------|
| Input | 187 tabular features | 187 tabular + NIfTI MRI scan |
| MRI processing | N/A | Automatic skull stripping + normalization |
| Heatmap | N/A | GradCAM++ NIfTI output for overlay |
| Brain output | N/A | Skull-stripped brain NIfTI |
| SHAP | 187 features | 187 tabular + MRI importance (188 total) |
| Checkpoints | Standard | Skull-stripped + deduped |

## Quick Start

### 1. Setup

```bash
# Option A: Automated setup (recommended)
bash setup.sh

# Option B: Manual setup
conda create -n mri_classifier_v2 python=3.10 -y
conda activate mri_classifier_v2
pip install -r requirements.txt
pip install -e nmed2024/
```

### 2. Run Prediction

```python
from api import MRICognitiveClassifierAPI

# Initialize (loads all 3 stage models)
api = MRICognitiveClassifierAPI()

# Predict with heatmap
result = api.predict(
    mri_path='patient_scan.nii.gz',
    features={'his_NACCAGE': 72, 'his_SEX': 1, 'bat_NACCMMSE': 22},
    output_dir='output/'
)

print(result['final_diagnosis'])      # e.g., 'AD', 'NC', 'MCI'
print(result['heatmaps'])             # Paths to NIfTI outputs
```

### 3. Run with SHAP Explanations

```python
result = api.predict_with_explanations(
    mri_path='patient_scan.nii.gz',
    features={'his_NACCAGE': 72, 'his_SEX': 1, 'bat_NACCMMSE': 22},
    output_dir='output/'
)

# Top contributing features
for feat in result['feature_importance']['top_features'][:5]:
    print(f"{feat['feature']}: {feat['shap_value']:+.4f}")
```

## Hierarchy

```
                Patient Input
               (MRI + 187 features)
                     |
               [Stage 1: Binary]
               DE vs Non-DE
              /              \
       Dementia            Non-Dementia
          |                     |
  [Stage 3: 9-class]    [Stage 2: 3-class]
  Dementia Subtypes      NC vs MCI vs iMCI
  AD, LBD, VD, FTD,
  PRD, NPH, SEF, PSY, ODE
```

## Input Format

### MRI Scan
- **Format**: `.nii` or `.nii.gz` (NIfTI)
- **Type**: T1-weighted structural MRI
- **Processing**: Automatic skull stripping, resize to 128x128x128, normalize

### Tabular Features
- **Format**: Python dict or JSON file with 187 NACC features
- **Missing data**: Set to `null` or `-4` -- model handles via attention masking
- **See**: `docs/feature_descriptions.csv` for complete feature list

## Output Format

### Prediction JSON
```json
{
    "final_diagnosis": "AD",
    "stage1": {"prediction": "DE", "de_probability": 0.82},
    "stage3": {"prediction": "AD", "probabilities": {"AD": 0.65, ...}},
    "heatmaps": {
        "heatmap_nifti": "output/heatmap.nii.gz",
        "brain_nifti": "output/brain.nii.gz",
        "overlay_nifti": "output/overlay.nii.gz"
    }
}
```

### NIfTI Outputs
| File | Description | Use |
|------|-------------|-----|
| `heatmap.nii.gz` | GradCAM++ attribution [0,1] | Overlay on brain with colormap |
| `brain.nii.gz` | Skull-stripped brain | Base image for overlay |
| `overlay.nii.gz` | Pre-blended 70/30 overlay | Quick visualization |

All three share the same affine matrix -- directly overlayable.

## Results (Skull-Stripped + Deduped)

| Stage | Task | AUROC |
|-------|------|-------|
| 1 | DE vs Non-DE | 0.9513 |
| 2 | NC/MCI/iMCI (avg) | 0.7598 |
| 3 | Dementia subtypes (avg) | 0.8247 |

## Performance

| Operation | Time | GPU Required |
|-----------|------|-------------|
| Inference (3 stages) | ~2s | Recommended |
| Heatmap (GradCAM++) | ~5s | Yes |
| SHAP explanations | ~3-5min | Yes |
| Skull stripping | ~10s | Yes |

## Command Line

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

## File Structure

```
mri_classifier_package_v2/
├── README.md                    # This file
├── api.py                       # Clean API for frontend
├── inference_pipeline.py        # Core pipeline
├── skull_strip.py               # deepbet wrapper
├── requirements.txt             # pip dependencies
├── environment.yml              # Conda environment
├── setup.sh / setup.bat         # Automated setup
├── models/
│   ├── stage1_de_vs_non_de.pt
│   ├── stage2_nc_mci_imci.pt
│   └── stage3_dementia_subtypes.pt
├── configs/
│   ├── stage1_de_mri_config.toml
│   ├── stage2_3way_mri_config.toml
│   └── stage3_10class_mri_config.toml
├── explainability/
│   ├── gradcam_mri.py
│   ├── improved_heatmap.py
│   ├── captum_shap_mri.py
│   ├── visualization.py
│   ├── reverse_transform.py
│   └── explain_mri.py
├── examples/
│   ├── example_inference.py
│   ├── example_with_shap.py
│   └── sample_features.json
├── docs/
│   ├── FRONTEND_API_GUIDE.md
│   ├── json_output_schema.md
│   └── feature_descriptions.csv
├── tests/
│   └── test_pipeline.py
└── nmed2024/                    # Bundled adrd package
```

## Troubleshooting

### CUDA not available
```python
# Force CPU (slower)
api = MRICognitiveClassifierAPI(device='cpu')
```

### Skull stripping fails
- Ensure `deepbet` is installed: `pip install deepbet`
- First run downloads the deepbet model (~100MB)
- Falls back to intensity-based masking if deepbet unavailable

### Missing features
Set missing values to `null` in JSON or `None` in Python dict. The model handles missing data via transformer attention masking.

### Import errors
```bash
pip install -e nmed2024/    # Install the bundled adrd package
```

## Citation

Based on the ADRDModel architecture from:
> Nature Medicine 2024 -- Transformer-based multimodal framework for
> differential diagnosis of cognitive impairment.
