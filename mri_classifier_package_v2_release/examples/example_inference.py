#!/usr/bin/env python3
"""
Example: Quick MRI + Tabular Inference

Demonstrates fast prediction with heatmap generation (no SHAP).
Inference takes ~7 seconds on GPU.
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
    # Replace these with your actual paths
    MRI_PATH = 'path/to/patient_scan.nii.gz'
    FEATURES_JSON = os.path.join(SCRIPT_DIR, 'sample_features.json')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

    # ==================== INITIALIZE ====================
    print("Loading models...")
    api = MRICognitiveClassifierAPI()
    print(f"Models loaded on device: {api.device}")

    # ==================== LOAD FEATURES ====================
    with open(FEATURES_JSON, 'r') as f:
        features = json.load(f)

    print(f"Loaded {len(features)} tabular features")

    # ==================== RUN PREDICTION ====================
    print("\nRunning inference...")
    result = api.predict(
        mri_path=MRI_PATH,
        features=features,
        output_dir=OUTPUT_DIR,
        generate_heatmap=True
    )

    # ==================== DISPLAY RESULTS ====================
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)

    # Stage 1
    s1 = result['stage1']
    print(f"\nStage 1 (Dementia Screening):")
    print(f"  Prediction: {s1['prediction']}")
    print(f"  DE Probability: {s1['de_probability']:.4f}")
    print(f"  Confidence: {s1['confidence']:.4f}")

    # Stage 2 (if Non-Dementia)
    if result['stage2']:
        s2 = result['stage2']
        print(f"\nStage 2 (Cognitive Status):")
        print(f"  Prediction: {s2['prediction']}")
        for cls, prob in s2['probabilities'].items():
            print(f"  {cls}: {prob:.4f}")

    # Stage 3 (if Dementia)
    if result['stage3']:
        s3 = result['stage3']
        print(f"\nStage 3 (Dementia Subtype):")
        print(f"  Prediction: {s3['prediction']}")
        # Show top 5 subtypes
        sorted_probs = sorted(s3['probabilities'].items(), key=lambda x: -x[1])
        for cls, prob in sorted_probs[:5]:
            print(f"  {cls}: {prob:.4f}")

    print(f"\nFINAL DIAGNOSIS: {result['final_diagnosis']}")

    # Heatmap outputs
    if result.get('heatmaps'):
        print(f"\nHeatmap outputs:")
        for key, path in result['heatmaps'].items():
            if isinstance(path, str) and path.endswith('.nii.gz'):
                print(f"  {key}: {path}")

    # Save full result
    result_path = os.path.join(OUTPUT_DIR, 'prediction_result.json')
    api.save_result(result, result_path)
    print(f"\nFull result saved to: {result_path}")


if __name__ == '__main__':
    main()
