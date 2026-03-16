@echo off
REM =============================================================================
REM MRI + Tabular Cognitive Classifier v2.0 - Setup Script (Windows)
REM =============================================================================
REM This script creates a conda environment and installs all dependencies.
REM Run this script once to set up the environment.
REM =============================================================================

echo ==============================================
echo Setting up MRI Classifier v2.0 Environment
echo ==============================================

REM Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: conda not found. Please install Miniconda or Anaconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

echo.
echo [1/5] Creating conda environment...
call conda create -n mri_classifier_v2 python=3.10 -y

echo.
echo [2/5] Activating environment...
call conda activate mri_classifier_v2

echo.
echo [3/5] Installing PyTorch with CUDA support...
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo [4/5] Installing dependencies...
pip install numpy pandas scipy scikit-learn tomli tqdm
pip install nibabel deepbet
pip install shap numba
pip install matplotlib pillow

echo.
echo [5/5] Installing nmed2024 package...
cd nmed2024
pip install -e .
cd ..

echo.
echo ==============================================
echo Setup Complete!
echo ==============================================
echo.
echo To activate the environment, run:
echo     conda activate mri_classifier_v2
echo.
echo Quick test:
echo     python -c "from api import MRICognitiveClassifierAPI; print('OK')"
echo.
echo For full usage, see README.md or docs/FRONTEND_API_GUIDE.md
