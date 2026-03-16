@echo off
REM =============================================================================
REM Hierarchical Classifier - Environment Setup Script (Windows)
REM =============================================================================
REM Run this in Anaconda Prompt to create the environment.
REM =============================================================================

echo ==============================================
echo Setting up Hierarchical Classifier Environment
echo ==============================================

where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: conda not found. Please install Miniconda first.
    echo Download: https://docs.conda.io/en/latest/miniconda.html
    exit /b 1
)

echo.
echo [1/4] Creating conda environment (Python 3.10)...
call conda create -n hierarchical_classifier python=3.10 -y

echo.
echo [2/4] Activating environment...
call conda activate hierarchical_classifier

echo.
echo [3/4] Installing Python dependencies...
pip install --upgrade pip
pip install torch torchvision
pip install numpy pandas scipy scikit-learn
pip install toml tqdm
pip install shap numba
pip install monai pillow icecream wandb
pip install flask flask-cors

echo.
echo [4/4] Installing nmed2024 package...
cd nmed2024
pip install -e .
cd ..

echo.
echo ==============================================
echo Environment Setup Complete!
echo ==============================================
echo.
echo To use: conda activate hierarchical_classifier
echo Then navigate to your model package and run tests.
echo.
pause
