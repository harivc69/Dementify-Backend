#!/usr/bin/env python3
"""
Example: Basic Inference with Hierarchical Cognitive Classifier

This script demonstrates how to:
1. Load the hierarchical classifier
2. Run inference on a single sample
3. Interpret the hierarchical results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_classifier import HierarchicalCognitiveClassifier


def main():
    print("=" * 70)
    print("Hierarchical Cognitive Impairment Classifier - Basic Example")
    print("=" * 70)

    # Initialize the classifier
    print("\n[1] Loading models...")
    classifier = HierarchicalCognitiveClassifier()

    # Get the list of required features
    features = classifier.get_feature_list()
    print(f"\n[2] Model expects {len(features)} input features")
    print(f"    Example features: {features[:5]}...")

    # Create a sample input (with some missing values)
    # In practice, you would load this from your data source
    print("\n[3] Creating sample input...")
    sample_data = {
        'his_NACCAGE': 75,           # Age
        'his_SEX': 2,                # Female
        'his_EDUC': 16,              # Years of education
        'bat_NACCMMSE': 22,          # MMSE score (mild impairment)
        'his_NACCFAM': 1,            # Family history of dementia
        # Most features are missing - model will use imputation
    }

    # Fill missing features with -4 (NACC missing code)
    for feat in features:
        if feat not in sample_data:
            sample_data[feat] = -4

    print(f"    Sample has {sum(1 for v in sample_data.values() if v != -4)} known features")
    print(f"    Sample has {sum(1 for v in sample_data.values() if v == -4)} missing features")

    # Run inference
    print("\n[4] Running hierarchical inference...")
    results = classifier.predict(sample_data)
    result = results[0]

    # Display results
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)

    # Stage 1 Result
    print(f"\n[STAGE 1] Dementia Screening:")
    print(f"    Prediction: {result['stage1']['prediction']}")
    print(f"    Confidence: {result['stage1']['confidence']:.1%}")
    print(f"    DE Probability: {result['stage1']['de_probability']:.3f}")

    # Stage 2 or Stage 3 based on Stage 1 result
    if result['stage2'] is not None:
        print(f"\n[STAGE 2] Cognitive Status (Non-Dementia):")
        print(f"    Prediction: {result['stage2']['prediction']}")
        print(f"    Full Name: {result['stage2']['prediction_full_name']}")
        print(f"    Confidence: {result['stage2']['confidence']:.1%}")
        if 'probabilities' in result['stage2']:
            print(f"    All Probabilities:")
            for cls, prob in result['stage2']['probabilities'].items():
                print(f"        {cls}: {prob:.1%}")

    if result['stage3'] is not None:
        print(f"\n[STAGE 3] Dementia Subtype:")
        print(f"    Prediction: {result['stage3']['prediction']}")
        print(f"    Full Name: {result['stage3']['prediction_full_name']}")
        print(f"    Confidence: {result['stage3']['confidence']:.1%}")
        print(f"    Top 3 Subtypes:")
        for item in result['stage3']['top_3']:
            print(f"        {item['subtype']}: {item['name']} ({item['probability']:.1%})")

    # Summary
    print(f"\n[SUMMARY]")
    print(f"    {result['summary']}")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
