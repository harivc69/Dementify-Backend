# Frontend Integration Guide

## Hierarchical Cognitive Impairment Classifier API

This guide provides everything needed to integrate the cognitive impairment classifier into your frontend application.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [API Reference](#api-reference)
4. [Input Formats](#input-formats)
5. [Output Formats](#output-formats)
6. [Response Schema](#response-schema)
7. [SHAP Explanations](#shap-explanations)
8. [Error Handling](#error-handling)
9. [Performance](#performance)
10. [Examples](#examples)

---

## Quick Start

```python
from api import CognitiveClassifierAPI

# Initialize (loads models once)
api = CognitiveClassifierAPI()

# Predict for a single patient
result = api.predict({
    'his_NACCAGE': 75,
    'bat_NACCMMSE': 22,
    # ... other features (missing features handled automatically)
})

print(result['summary'])
# Output: "DEMENTIA (83.7%) - Primary Subtype: Alzheimer's Disease (72.1%)"

# Get with SHAP explanations (slower, ~5 min)
result = api.predict_with_explanations(patient_data)
print(result['top_contributing_features'])
```

---

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install the nmed2024 package
pip install -e ../nmed2024/

# 3. Verify installation
python -c "from api import CognitiveClassifierAPI; print('OK')"
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- ~2GB disk space for models

---

## API Reference

### `CognitiveClassifierAPI`

Main class for all predictions.

#### Constructor

```python
api = CognitiveClassifierAPI(
    device='cuda',   # 'cuda' or 'cpu' (auto-detected if None)
    verbose=True     # Print loading messages
)
```

#### Methods

| Method | Description | Speed |
|--------|-------------|-------|
| `predict()` | Standard prediction | ~100ms/patient |
| `predict_with_explanations()` | Prediction + SHAP | ~5min/patient |
| `predict_stage1_only()` | Dementia screening only | ~50ms/patient |
| `validate_input()` | Check input data | Instant |
| `save_results()` | Save to file | Instant |

---

## Input Formats

The API accepts multiple input formats:

### 1. Python Dictionary (Single Patient)

```python
patient = {
    'his_NACCAGE': 75,
    'his_SEX': 2,
    'bat_NACCMMSE': 22,
    # Missing features: set to None or omit entirely
}
result = api.predict(patient)
```

### 2. List of Dictionaries (Batch)

```python
patients = [
    {'his_NACCAGE': 75, 'bat_NACCMMSE': 22},
    {'his_NACCAGE': 68, 'bat_NACCMMSE': 28},
]
results = api.predict(patients)  # Returns list
```

### 3. Pandas DataFrame

```python
import pandas as pd
df = pd.DataFrame(patients)
results = api.predict(df)
```

### 4. CSV File

```python
results = api.predict('patients.csv')
```

### 5. JSON File

```python
results = api.predict('patients.json')
```

### Handling Missing Data

The model automatically handles missing data. You can represent missing values as:
- `None` or `null`
- Empty string `""`
- `NaN`
- NACC missing codes: `-4`, `8`, `9`, `88`, `99`, `888`, `999`, `8888`, `9999`

```python
# All of these are treated as missing:
patient = {
    'his_NACCAGE': 75,       # Valid value
    'bat_NACCMMSE': None,    # Missing
    'his_SEX': -4,           # NACC missing code
}
```

---

## Output Formats

Specify output format with the `output_format` parameter:

```python
# Python dict (default)
result = api.predict(patient, output_format='dict')

# JSON string
json_str = api.predict(patient, output_format='json')

# Pandas DataFrame
df = api.predict(patients, output_format='dataframe')

# CSV string
csv_str = api.predict(patients, output_format='csv')
```

---

## Response Schema

### Standard Prediction Response

```json
{
  "sample_id": 0,
  "original_id": "PATIENT_123",

  "stage1": {
    "prediction": "Dementia",
    "confidence": 0.837,
    "de_probability": 0.837,
    "probabilities": {
      "Dementia": 0.837,
      "Non-Dementia": 0.163
    }
  },

  "stage2": null,

  "stage3": {
    "prediction": "AD",
    "prediction_full_name": "Alzheimer's Disease",
    "confidence": 0.721,
    "top_3": [
      {"subtype": "AD", "name": "Alzheimer's Disease", "probability": 0.721},
      {"subtype": "VD", "name": "Vascular Dementia", "probability": 0.452},
      {"subtype": "LBD", "name": "Lewy Body Dementia", "probability": 0.318}
    ],
    "all_probabilities": {
      "AD": 0.721, "LBD": 0.318, "VD": 0.452, "FTD": 0.156,
      "PRD": 0.023, "NPH": 0.089, "SEF": 0.134, "PSY": 0.267,
      "TBI": 0.045, "ODE": 0.078
    }
  },

  "summary": "DEMENTIA (83.7%) - Primary Subtype: Alzheimer's Disease (72.1%)"
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | int | Index in batch (0-based) |
| `original_id` | string | Patient ID from input (if provided) |
| `stage1` | object | Dementia screening result |
| `stage2` | object/null | Non-dementia classification (null if dementia) |
| `stage3` | object/null | Dementia subtype (null if non-dementia) |
| `summary` | string | Human-readable summary |

### Stage 1 Fields

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "Dementia" or "Non-Dementia" |
| `confidence` | float | Confidence in prediction (0-1) |
| `de_probability` | float | Probability of dementia (0-1) |
| `probabilities` | object | Both class probabilities |

### Stage 2 Fields (Non-Dementia)

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "NC", "MCI", or "IMCI" |
| `prediction_full_name` | string | Full name of condition |
| `confidence` | float | Confidence (0-1) |
| `probabilities` | object | All class probabilities |

**Class Definitions:**
- **NC**: Normal Cognition
- **MCI**: Mild Cognitive Impairment
- **IMCI**: MCI with Functional Impairment

### Stage 3 Fields (Dementia)

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | Subtype code (e.g., "AD") |
| `prediction_full_name` | string | Full name |
| `confidence` | float | Confidence (0-1) |
| `top_3` | array | Top 3 subtypes with probabilities |
| `all_probabilities` | object | All 10 subtype probabilities |

**Subtype Codes:**
| Code | Full Name |
|------|-----------|
| AD | Alzheimer's Disease |
| LBD | Lewy Body Dementia |
| VD | Vascular Dementia |
| FTD | Frontotemporal Dementia |
| PRD | Prion Disease (CJD) |
| NPH | Normal Pressure Hydrocephalus |
| SEF | Systemic/Environmental Factors |
| PSY | Psychiatric Causes |
| TBI | Traumatic Brain Injury |
| ODE | Other Dementia Etiologies |

---

## SHAP Explanations

SHAP (SHapley Additive exPlanations) provides interpretable explanations showing which features contributed most to the prediction.

### Usage

```python
result = api.predict_with_explanations(
    patient,
    n_top_features=10  # Number of top features to return
)
```

### Response with SHAP

```json
{
  "sample_id": 0,
  "stage1": {
    "prediction": "Dementia",
    "confidence": 0.837,
    "explanation": {
      "top_features": [
        {
          "feature": "bat_NACCMMSE",
          "feature_description": "Cognitive Test: NACCMMSE",
          "shap_value": 0.245,
          "direction": "increases",
          "importance": 0.245
        },
        {
          "feature": "his_NACCAGE",
          "feature_description": "Medical History: NACCAGE",
          "shap_value": 0.182,
          "direction": "increases",
          "importance": 0.182
        }
      ]
    }
  },
  "stage3": {
    "prediction": "AD",
    "explanation": {
      "top_features": [...]
    }
  },
  "top_contributing_features": [
    {
      "feature": "bat_NACCMMSE",
      "feature_description": "Cognitive Test: NACCMMSE",
      "shap_value": 0.245,
      "direction": "increases",
      "importance": 0.245
    }
  ],
  "summary": "..."
}
```

### SHAP Field Descriptions

| Field | Description |
|-------|-------------|
| `feature` | Feature code name |
| `feature_description` | Human-readable description |
| `shap_value` | SHAP contribution value |
| `direction` | "increases" or "decreases" prediction |
| `importance` | Absolute SHAP value |

### Performance Note

SHAP computation is expensive (~5 minutes per patient on GPU). For batch processing:
- Use `predict()` for fast inference
- Use `predict_with_explanations()` only for selected patients

---

## Error Handling

### Common Errors

```python
# File not found
try:
    result = api.predict('nonexistent.csv')
except FileNotFoundError as e:
    print(f"File error: {e}")

# Invalid format
try:
    result = api.predict(patient, output_format='invalid')
except ValueError as e:
    print(f"Format error: {e}")
```

### Input Validation

```python
report = api.validate_input(patient_data)

print(report)
# {
#   'valid': True,
#   'total_features': 187,
#   'provided_features': 5,
#   'missing_features': 182,
#   'unknown_features': [],
#   'avg_non_null_features': 5.0,
#   'warnings': ['182 expected features missing (will use imputation)']
# }
```

---

## Performance

### Benchmarks (NVIDIA GPU)

| Operation | Time per Patient |
|-----------|------------------|
| Standard prediction | ~100ms |
| Stage 1 only | ~50ms |
| With SHAP | ~5 minutes |

### Optimization Tips

1. **Reuse API instance**: Model loading takes ~10 seconds
2. **Batch processing**: More efficient than single predictions
3. **GPU acceleration**: 3-5x faster than CPU

```python
# Good: Reuse API instance
api = CognitiveClassifierAPI()
for patient in patients:
    result = api.predict(patient)

# Better: Batch processing
results = api.predict(patients)  # All at once
```

---

## Examples

### Example 1: Single Patient Prediction

```python
from api import CognitiveClassifierAPI

api = CognitiveClassifierAPI()

patient = {
    'his_NACCAGE': 78,
    'his_SEX': 1,
    'his_EDUC': 12,
    'bat_NACCMMSE': 18,
    'original_id': 'PATIENT_001'
}

result = api.predict(patient)

print(f"Prediction: {result['stage1']['prediction']}")
print(f"Confidence: {result['stage1']['confidence']:.1%}")
print(f"Summary: {result['summary']}")

if result['stage3']:
    print(f"Subtype: {result['stage3']['prediction_full_name']}")
```

### Example 2: Batch Processing from CSV

```python
api = CognitiveClassifierAPI()

# Load and predict
results = api.predict('patients.csv', output_format='dataframe')

# Analyze results
dementia_count = (results['stage1_prediction'] == 'Dementia').sum()
print(f"Dementia cases: {dementia_count}/{len(results)}")

# Save results
api.save_results(results.to_dict('records'), 'results.json')
```

### Example 3: With SHAP Explanations

```python
api = CognitiveClassifierAPI()

result = api.predict_with_explanations(patient, n_top_features=5)

print("Top Contributing Features:")
for feat in result['top_contributing_features']:
    direction = "+" if feat['shap_value'] > 0 else "-"
    print(f"  {feat['feature_description']}: {direction}{abs(feat['shap_value']):.3f}")
```

### Example 4: REST API Integration

```python
from flask import Flask, request, jsonify
from api import CognitiveClassifierAPI

app = Flask(__name__)
api = CognitiveClassifierAPI(verbose=False)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = api.predict(data)
    return jsonify(result)

@app.route('/predict_with_shap', methods=['POST'])
def predict_with_shap():
    data = request.json
    result = api.predict_with_explanations(data)
    return jsonify(result)
```

---

## Feature List

The model expects 187 features. Get the full list:

```python
api = CognitiveClassifierAPI()
features = api.feature_list
print(f"Total features: {len(features)}")  # 187
```

Key feature prefixes:
- `his_`: Demographics/medical history
- `bat_`: Cognitive test scores (MMSE, MoCA, etc.)
- `med_`: Medications
- `exam_`: Neurological exam findings
- `faq_`: Functional assessment

---

## Support

For issues or questions:
1. Check the test suite: `python tests/test_comprehensive.py`
2. Review example scripts in `examples/`
3. Contact the UIUC research team
