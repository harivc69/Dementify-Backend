# Hierarchical Classifier - Environment Setup

This folder contains everything needed to set up the Python environment for the Hierarchical Cognitive Impairment Classifier.

## Prerequisites

- **Conda** (Miniconda or Anaconda) - [Download here](https://docs.conda.io/en/latest/miniconda.html)

## Quick Setup

### Linux / macOS

```bash
cd hierarchical_classifier_env_setup
chmod +x setup.sh
./setup.sh
```

### Windows

1. Open **Anaconda Prompt** (not regular Command Prompt)
2. Navigate to this folder
3. Run: `setup.bat`

## After Setup

1. Activate the environment:
   ```bash
   conda activate hierarchical_classifier
   ```

2. Navigate to your model package folder:
   ```bash
   cd /path/to/hierarchical_model_package
   ```

3. Verify installation:
   ```bash
   python -c "from api import CognitiveClassifierAPI; api = CognitiveClassifierAPI(); print('Success!')"
   ```

4. Run tests:
   ```bash
   python tests/test_comprehensive.py --quick
   ```

## Folder Contents

```
hierarchical_classifier_env_setup/
├── setup.sh                    # Setup script (Linux/macOS)
├── setup.bat                   # Setup script (Windows)
├── README.md                   # This file
├── environment.yml             # Conda environment spec
├── environment_requirements.txt # Pinned pip requirements
└── nmed2024/                   # Required dependency package
    ├── adrd/                   # Core ADRD model code
    ├── setup.py                # Package installer
    └── dev/data/               # Config files
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `conda: command not found` | Install Miniconda from link above |
| `ModuleNotFoundError: No module named 'adrd'` | Re-run `pip install -e nmed2024/` |
| Package import errors | Make sure environment is activated: `conda activate hierarchical_classifier` |

## Installed Packages

- Python 3.10
- PyTorch (with CUDA support if available)
- NumPy, Pandas, SciPy, scikit-learn
- SHAP (for explainability)
- MONAI (medical imaging)
- Flask (for API server examples)
