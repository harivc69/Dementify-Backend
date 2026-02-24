#!/usr/bin/env python3
"""
Example: Inference with SHAP Explanations

This script demonstrates how to:
1. Load the hierarchical classifier
2. Run inference with SHAP explanations
3. Interpret feature contributions for each prediction

SHAP (SHapley Additive exPlanations) provides interpretable explanations
by showing which features contributed most to the model's prediction.

Note: SHAP computation is slower than standard inference (~30-60 seconds per sample).
For batch processing, consider using predict() for fast inference and
predict_with_explanations() for selected samples that need interpretation.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_classifier import HierarchicalCognitiveClassifier


def main():
    print("=" * 70)
    print("Hierarchical Classifier with SHAP Explanations")
    print("=" * 70)

    # Initialize the classifier
    print("\n[1] Loading models...")
    classifier = HierarchicalCognitiveClassifier()

    # Load sample data
    print("\n[2] Loading sample data...")
    sample_path = Path(__file__).parent / "sample_data.json"

    if sample_path.exists():
        with open(sample_path, 'r') as f:
            samples = json.load(f)
        # Use first sample for demo
        sample = samples[0]
        print(f"    Loaded sample with original_id: {sample.get('original_id', 'N/A')}")
    else:
        # Create a minimal sample
        features = classifier.get_feature_list()
        sample = {feat: -4 for feat in features}  # All missing
        sample['his_NACCAGE'] = 78
        sample['bat_NACCMMSE'] = 18  # Low MMSE suggests impairment
        print("    Created synthetic sample with minimal features")

    # Run inference WITH SHAP explanations
    print("\n[3] Running inference with SHAP explanations...")
    print("    (This may take 30-60 seconds per sample)")

    results = classifier.predict_with_explanations(
        sample,
        n_top_features=10,
        compute_shap=True
    )
    result = results[0]

    # Display results
    print("\n" + "=" * 70)
    print("PREDICTION RESULTS")
    print("=" * 70)

    # Stage 1 Results
    print(f"\n[STAGE 1] Dementia Screening:")
    print(f"    Prediction: {result['stage1']['prediction']}")
    print(f"    Confidence: {result['stage1']['confidence']:.1%}")

    if 'explanation' in result['stage1'] and 'top_features' in result['stage1']['explanation']:
        print(f"\n    Top Contributing Features (Stage 1):")
        for i, feat in enumerate(result['stage1']['explanation']['top_features'][:5], 1):
            direction = "+" if feat['shap_value'] > 0 else "-"
            print(f"    {i}. {feat['feature_description']}")
            print(f"       SHAP: {direction}{abs(feat['shap_value']):.4f} ({feat['direction']} dementia probability)")

    # Stage 2 or Stage 3 results
    if result['stage2'] is not None:
        print(f"\n[STAGE 2] Cognitive Status Classification:")
        print(f"    Prediction: {result['stage2']['prediction_full_name']}")
        print(f"    Confidence: {result['stage2']['confidence']:.1%}")

        if 'explanation' in result['stage2'] and 'top_features' in result['stage2']['explanation']:
            print(f"\n    Top Contributing Features (Stage 2):")
            for i, feat in enumerate(result['stage2']['explanation']['top_features'][:5], 1):
                direction = "+" if feat['shap_value'] > 0 else "-"
                print(f"    {i}. {feat['feature_description']}")
                print(f"       SHAP: {direction}{abs(feat['shap_value']):.4f}")

    if result['stage3'] is not None:
        print(f"\n[STAGE 3] Dementia Subtype Classification:")
        print(f"    Prediction: {result['stage3']['prediction_full_name']}")
        print(f"    Confidence: {result['stage3']['confidence']:.1%}")
        print(f"    Top 3 Subtypes:")
        for item in result['stage3']['top_3']:
            print(f"        {item['subtype']}: {item['name']} ({item['probability']:.1%})")

        if 'explanation' in result['stage3'] and 'top_features' in result['stage3']['explanation']:
            print(f"\n    Top Contributing Features (Stage 3):")
            for i, feat in enumerate(result['stage3']['explanation']['top_features'][:5], 1):
                direction = "+" if feat['shap_value'] > 0 else "-"
                print(f"    {i}. {feat['feature_description']}")
                print(f"       SHAP: {direction}{abs(feat['shap_value']):.4f}")

    # Overall top contributing features
    if 'top_contributing_features' in result:
        print(f"\n" + "=" * 70)
        print("OVERALL TOP CONTRIBUTING FEATURES")
        print("=" * 70)
        for i, feat in enumerate(result['top_contributing_features'], 1):
            print(f"{i}. {feat['feature_description']}")
            print(f"   Feature: {feat['feature']}")
            print(f"   SHAP Value: {feat['shap_value']:.4f}")
            print(f"   Effect: {feat['direction']} prediction probability")
            print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{result['summary']}")

    # Save full results to JSON
    output_path = Path(__file__).parent / "shap_output_example.json"

    # Create a serializable version (remove full_shap for brevity)
    result_summary = {
        'sample_id': result['sample_id'],
        'original_id': result.get('original_id'),
        'stage1': {
            'prediction': result['stage1']['prediction'],
            'confidence': result['stage1']['confidence'],
            'top_features': result['stage1'].get('explanation', {}).get('top_features', [])[:5]
        },
        'stage2': None,
        'stage3': None,
        'summary': result['summary'],
        'top_contributing_features': result.get('top_contributing_features', [])
    }

    if result['stage2']:
        result_summary['stage2'] = {
            'prediction': result['stage2']['prediction'],
            'prediction_full_name': result['stage2']['prediction_full_name'],
            'confidence': result['stage2']['confidence'],
            'top_features': result['stage2'].get('explanation', {}).get('top_features', [])[:5]
        }

    if result['stage3']:
        result_summary['stage3'] = {
            'prediction': result['stage3']['prediction'],
            'prediction_full_name': result['stage3']['prediction_full_name'],
            'confidence': result['stage3']['confidence'],
            'top_3': result['stage3']['top_3'],
            'top_features': result['stage3'].get('explanation', {}).get('top_features', [])[:5]
        }

    with open(output_path, 'w') as f:
        json.dump(result_summary, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
