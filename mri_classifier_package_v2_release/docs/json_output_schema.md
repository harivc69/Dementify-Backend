# JSON Output Schema

## `predict()` Response

```json
{
    "timestamp": "2026-03-05T14:30:00.123456",
    "mri_path": "string (path to input NIfTI)",

    "stage1": {
        "prediction": "DE | Non-DE",
        "de_probability": "float [0, 1]",
        "confidence": "float [0, 1]"
    },

    "stage2": "null (if Dementia) | object (if Non-Dementia)",
    "stage2_when_present": {
        "prediction": "NC | MCI | IMCI",
        "probabilities": {
            "NC": "float [0, 1]",
            "MCI": "float [0, 1]",
            "IMCI": "float [0, 1]"
        },
        "confidence": "float [0, 1]"
    },

    "stage3": "null (if Non-Dementia) | object (if Dementia)",
    "stage3_when_present": {
        "prediction": "AD | LBD | VD | FTD | PRD | NPH | SEF | PSY | ODE",
        "probabilities": {
            "AD": "float [0, 1]",
            "LBD": "float [0, 1]",
            "VD": "float [0, 1]",
            "FTD": "float [0, 1]",
            "PRD": "float [0, 1]",
            "NPH": "float [0, 1]",
            "SEF": "float [0, 1]",
            "PSY": "float [0, 1]",
            "ODE": "float [0, 1]"
        },
        "confidence": "float [0, 1]"
    },

    "final_diagnosis": "string (terminal prediction: NC, MCI, IMCI, AD, LBD, etc.)",

    "heatmaps": {
        "heatmap_nifti": "string (path to heatmap.nii.gz)",
        "brain_nifti": "string (path to brain.nii.gz)",
        "overlay_nifti": "string (path to overlay.nii.gz)",
        "stage": "stage1 | stage3",
        "target_class": "int",
        "method": "gradcam_pp"
    }
}
```

## `predict_with_explanations()` Response

Same as `predict()` plus:

```json
{
    "... all predict() fields ...",

    "stage1": {
        "... prediction fields ...",
        "explanation": {
            "top_features": [
                {
                    "feature": "string (feature name)",
                    "description": "string (human-readable category: name)",
                    "shap_value": "float (signed SHAP value)",
                    "direction": "increases | decreases",
                    "importance": "float (|shap_value|)"
                }
            ],
            "mri_importance": "float [0, 1]"
        }
    },

    "feature_importance": {
        "tabular_shap": {
            "feature_names": ["string", "..."],
            "shap_values": ["float", "..."],
            "base_value": "float",
            "target": "string (target label, e.g. 'DE')"
        },
        "mri_importance": "float [0, 1] (mean GradCAM++ activation over brain voxels)",
        "top_features": [
            {
                "feature": "string",
                "description": "string",
                "shap_value": "float",
                "direction": "increases | decreases",
                "importance": "float"
            }
        ]
    }
}
```

## Diagnosis Codes

### Stage 1

| Code | Meaning |
|------|---------|
| `DE` | Dementia |
| `Non-DE` | Non-Dementia |

### Stage 2 (Non-Dementia path)

| Code | Meaning |
|------|---------|
| `NC` | Normal Cognition |
| `MCI` | Mild Cognitive Impairment |
| `IMCI` | MCI with Functional Impairment |

### Stage 3 (Dementia path)

| Code | Full Name |
|------|-----------|
| `AD` | Alzheimer's Disease |
| `LBD` | Lewy Body Dementia |
| `VD` | Vascular Dementia |
| `FTD` | Frontotemporal Dementia |
| `PRD` | Prion Disease (CJD) |
| `NPH` | Normal Pressure Hydrocephalus |
| `SEF` | Systemic/Environmental Factors |
| `PSY` | Psychiatric Causes |
| `ODE` | Other Dementia Etiologies |

## NIfTI Output Files

| File | Shape | Data Type | Range | Description |
|------|-------|-----------|-------|-------------|
| `heatmap.nii.gz` | Same as input MRI | float32 | [0, 1] | GradCAM++ attribution map |
| `brain.nii.gz` | Same as input MRI | float32 | Raw intensity | Skull-stripped brain |
| `overlay.nii.gz` | Same as input MRI | float32 | [0, 1] | Pre-blended 70/30 overlay |

All three files share the same **affine matrix** as the input MRI, allowing direct overlay.
