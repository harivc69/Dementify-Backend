# Hierarchical Cognitive Impairment Classifier

A production-ready 3-stage deep learning model for differential diagnosis of cognitive impairment using 187 tabular clinical features.

---

## Package Contents

```
hierarchical_model_package/
│
├── api.py                      # Main API - USE THIS FOR INTEGRATION
├── hierarchical_classifier.py  # Core classifier implementation
├── requirements.txt            # Python dependencies
│
├── models/                     # Trained model checkpoints (22 MB)
│   ├── stage1_de_vs_nonde.pt      # Stage 1: Dementia screening
│   ├── stage2_nc_mci_imci.pt      # Stage 2: Cognitive status
│   └── stage3_dementia_subtypes.pt # Stage 3: 10 dementia subtypes
│
├── configs/                    # Feature configuration files
│   ├── stage1_de_config.toml      # 187 features for Stage 1
│   ├── stage2_3way_config.toml    # 187 features for Stage 2
│   └── stage3_10class_config.toml # 187 features for Stage 3
│
├── examples/                   # Example scripts
│   ├── example_inference.py       # Basic single-patient example
│   ├── example_with_shap.py       # SHAP explanations example
│   ├── batch_inference.py         # Batch processing from files
│   ├── api_server.py              # Flask REST API example
│   └── sample_data.json           # Sample patient data (3 patients)
│
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_comprehensive.py      # 36 comprehensive tests
│
└── docs/                       # Documentation
    ├── FRONTEND_API_GUIDE.md      # Complete integration guide
    └── MODEL_SPECIFICATIONS.pdf   # Detailed performance metrics
```

---

## Quick Start

```python
from api import CognitiveClassifierAPI

# Initialize once (loads models, ~10 seconds)
api = CognitiveClassifierAPI()

# Predict for a patient
result = api.predict({
    'his_NACCAGE': 75,
    'bat_NACCMMSE': 22,
    # Other features optional - missing data handled automatically
})

print(result['summary'])
# "DEMENTIA (83.7%) - Primary Subtype: Alzheimer's Disease (72.1%)"
```

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install the nmed2024 package (required)
pip install -e ../nmed2024/

# Verify installation
python -c "from api import CognitiveClassifierAPI; api = CognitiveClassifierAPI(); print('OK')"
```

---

## API Methods

### `predict(data, output_format='dict')`
Standard hierarchical prediction (~100ms per patient)

```python
# Single patient
result = api.predict({'his_NACCAGE': 75, 'bat_NACCMMSE': 22})

# Batch from file
results = api.predict('patients.csv')

# Different output formats
json_str = api.predict(data, output_format='json')
df = api.predict(data, output_format='dataframe')
```

### `predict_with_explanations(data, n_top_features=20)`
Prediction with SHAP feature importance (~5 min per patient)

```python
result = api.predict_with_explanations(patient_data, n_top_features=50)

# Access top contributing features
for feat in result['top_contributing_features']:
    print(f"{feat['feature']}: {feat['shap_value']:.3f} ({feat['direction']})")

# Get ALL 187 features ranked by importance
result = api.predict_with_explanations(patient_data, n_top_features=187)
```

### `predict_stage1_only(data)`
Fast dementia screening only (~50ms per patient)

```python
result = api.predict_stage1_only(patient_data)
# Returns only Stage 1 results (no Stage 2/3)
```

---

## Input Formats

The API accepts multiple input formats:

| Format | Example |
|--------|---------|
| **Dict** | `{'his_NACCAGE': 75, 'bat_NACCMMSE': 22}` |
| **List** | `[patient1, patient2, ...]` |
| **DataFrame** | `pd.DataFrame(patients)` |
| **CSV file** | `'patients.csv'` |
| **JSON file** | `'patients.json'` |

### Missing Data

Missing values are handled automatically. Use any of:
- `None` or `null`
- Empty string
- `NaN`
- NACC codes: `-4`, `8`, `9`, `88`, `99`, `888`, `999`

---

## Output Format

```json
{
  "sample_id": 0,
  "original_id": "PATIENT_123",

  "stage1": {
    "prediction": "Dementia",
    "confidence": 0.837,
    "de_probability": 0.837,
    "probabilities": {"Dementia": 0.837, "Non-Dementia": 0.163}
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

  "top_contributing_features": [
    {"feature": "bat_NACCMMSE", "shap_value": 0.245, "direction": "increases", "importance": 0.245},
    {"feature": "his_NACCAGE", "shap_value": 0.182, "direction": "increases", "importance": 0.182}
  ],

  "summary": "DEMENTIA (83.7%) - Primary Subtype: Alzheimer's Disease (72.1%)"
}
```

---

## Model Architecture

```
Patient Data (187 features)
         │
         ▼
┌─────────────────────────────────┐
│  Stage 1: Dementia Screening    │  AUC-ROC: 0.9527
│  (Dementia vs Non-Dementia)     │
└─────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
Non-DE      Dementia
    │         │
    ▼         ▼
┌────────┐  ┌────────────────────┐
│Stage 2 │  │     Stage 3        │
│NC/MCI/ │  │  10 Subtypes:      │
│iMCI    │  │  AD, LBD, VD, FTD  │
│        │  │  PRD, NPH, SEF,    │
│AUC:0.77│  │  PSY, TBI, ODE     │
└────────┘  │  AUC: 0.78         │
            └────────────────────┘
```

---

## Performance

| Stage | Task | AUC-ROC | AUC-PR |
|-------|------|---------|--------|
| Stage 1 | Dementia vs Non-Dementia | **0.9527** | 0.8180 |
| Stage 2 | NC / MCI / iMCI | 0.7667 | 0.5478 |
| Stage 3 | 10 Dementia Subtypes | 0.7754 | 0.2937 |

### Speed Benchmarks

| Operation | Time |
|-----------|------|
| Model loading | ~10 seconds |
| Standard prediction | ~100ms / patient |
| With SHAP | ~5 min / patient |

---

## Testing

```bash
# Run all 36 tests
python tests/test_comprehensive.py

# Quick test (3 tests)
python tests/test_comprehensive.py --quick

# Include SHAP test (~5 min)
python tests/test_comprehensive.py --shap
```

---

## File Descriptions

| File | Purpose |
|------|---------|
| `api.py` | **Main entry point** - Clean API for frontend integration |
| `hierarchical_classifier.py` | Core model implementation with SHAP support |
| `requirements.txt` | Python package dependencies |
| `models/*.pt` | Trained PyTorch model checkpoints |
| `configs/*.toml` | Feature definitions (187 features per stage) |
| `examples/` | Working example scripts |
| `tests/` | Comprehensive test suite |
| `docs/FRONTEND_API_GUIDE.md` | Detailed integration documentation |
| `docs/MODEL_SPECIFICATIONS.pdf` | Full performance metrics and architecture |

---

## Documentation

1. **This README** - Quick start and overview
2. **docs/FRONTEND_API_GUIDE.md** - Complete integration guide with examples
3. **docs/MODEL_SPECIFICATIONS.pdf** - Detailed performance metrics by class

---

## Support

For issues:
1. Run tests: `python tests/test_comprehensive.py`
2. Check examples in `examples/` directory
3. Review `docs/FRONTEND_API_GUIDE.md`

---

## License

Research use only. Contact UIUC for licensing inquiries.
