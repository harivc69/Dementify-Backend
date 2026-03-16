#!/usr/bin/env python3
"""
Example: Full Inference with SHAP Explanations

Demonstrates prediction with SHAP tabular explanations + MRI heatmap.
Takes ~3-5 minutes per patient due to SHAP computation.
"""

import os
import sys
import json

# Add package root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PACKAGE_DIR)

from api import MRICognitiveClassifierAPI


def main():
    # ==================== CONFIGURATION ====================
    MRI_PATH = 'path/to/patient_scan.nii.gz'
    FEATURES_JSON = os.path.join(SCRIPT_DIR, 'sample_features.json')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_shap')

    # ==================== INITIALIZE ====================
    print("Loading models...")
    api = MRICognitiveClassifierAPI()

    # ==================== LOAD FEATURES ====================
    with open(FEATURES_JSON, 'r') as f:
        features = json.load(f)

    # ==================== RUN WITH EXPLANATIONS ====================
    print("\nRunning inference with SHAP explanations...")
    print("(This will take ~3-5 minutes)")

    result = api.predict_with_explanations(
        mri_path=MRI_PATH,
        features=features,
        output_dir=OUTPUT_DIR,
        n_top_features=20
    )

    # ==================== DISPLAY RESULTS ====================
    print("\n" + "=" * 50)
    print("PREDICTION WITH EXPLANATIONS")
    print("=" * 50)

    print(f"\nFinal Diagnosis: {result['final_diagnosis']}")
    print(f"DE Probability: {result['stage1']['de_probability']:.4f}")

    # Feature importance
    fi = result.get('feature_importance', {})

    print(f"\nMRI Importance: {fi.get('mri_importance', 'N/A'):.4f}")

    top_features = fi.get('top_features', [])
    if top_features:
        print(f"\nTop {len(top_features)} Contributing Features:")
        print("-" * 60)
        print(f"{'Feature':<25} {'SHAP Value':>12} {'Direction':<12} {'Importance':>10}")
        print("-" * 60)
        for feat in top_features:
            print(f"{feat['feature']:<25} {feat['shap_value']:>+12.4f} "
                  f"{feat['direction']:<12} {feat['importance']:>10.4f}")

    # Heatmap outputs
    if result.get('heatmaps'):
        print(f"\nHeatmap outputs:")
        for key, path in result['heatmaps'].items():
            if isinstance(path, str) and path.endswith('.nii.gz'):
                print(f"  {key}: {path}")

    # Save full result
    result_path = os.path.join(OUTPUT_DIR, 'prediction_with_shap.json')
    api.save_result(result, result_path)
    print(f"\nFull result saved to: {result_path}")

    # ==================== VERIFY OUTPUTS ====================
    print("\n" + "=" * 50)
    print("OUTPUT VERIFICATION")
    print("=" * 50)

    import nibabel as nib

    for name in ['heatmap.nii.gz', 'brain.nii.gz', 'overlay.nii.gz']:
        fpath = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(fpath):
            img = nib.load(fpath)
            print(f"  {name}: shape={img.shape}, dtype={img.get_data_dtype()}")
        else:
            print(f"  {name}: NOT FOUND")


if __name__ == '__main__':
    main()
