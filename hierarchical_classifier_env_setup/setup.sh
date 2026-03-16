#!/bin/bash
# =============================================================================
# Hierarchical Classifier - Environment Setup Script
# =============================================================================
# Run this script to create the conda environment with all dependencies.
# After running this, you can use it with the model package.
# =============================================================================

set -e

echo "=============================================="
echo "Setting up Hierarchical Classifier Environment"
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda first."
    echo "Download: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo ""
echo "[1/4] Creating conda environment (Python 3.10)..."
conda create -n hierarchical_classifier python=3.10 -y

echo ""
echo "[2/4] Activating environment..."
eval "$(conda shell.bash hook)"
conda activate hierarchical_classifier

echo ""
echo "[3/4] Installing Python dependencies..."
pip install --upgrade pip

# Core ML
pip install torch torchvision

# Data processing
pip install numpy pandas scipy scikit-learn

# Configuration & utilities
pip install toml tqdm

# Explainability
pip install shap numba

# Medical imaging & experiment tracking
pip install monai pillow icecream wandb

# Web API (optional, for examples)
pip install flask flask-cors

echo ""
echo "[4/4] Installing nmed2024 package..."
cd "$SCRIPT_DIR/nmed2024"
pip install -e .
cd "$SCRIPT_DIR"

echo ""
echo "=============================================="
echo "Environment Setup Complete!"
echo "=============================================="
echo ""
echo "To use the environment:"
echo "    conda activate hierarchical_classifier"
echo ""
echo "Then navigate to your model package folder and run:"
echo "    python tests/test_comprehensive.py --quick"
echo ""
