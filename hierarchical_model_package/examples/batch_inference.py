#!/usr/bin/env python3
"""
Example: Batch Inference from CSV or JSON Files

This script demonstrates how to:
1. Load patient data from CSV or JSON files
2. Run batch inference on multiple samples
3. Optionally compute SHAP explanations for interpretability
4. Save results to JSON or CSV format

Usage:
    python batch_inference.py input.csv output.json
    python batch_inference.py input.json output.csv
    python batch_inference.py input.csv output.json --shap  # With SHAP explanations
"""

import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hierarchical_classifier import HierarchicalCognitiveClassifier


def load_input_data(filepath: str) -> pd.DataFrame:
    """Load input data from CSV or JSON file."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    if filepath.suffix.lower() == '.csv':
        print(f"Loading CSV file: {filepath}")
        return pd.read_csv(filepath)
    elif filepath.suffix.lower() == '.json':
        print(f"Loading JSON file: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def save_results(results: list, filepath: str, input_data: pd.DataFrame = None, include_shap: bool = False):
    """Save inference results to JSON or CSV file."""
    filepath = Path(filepath)

    if filepath.suffix.lower() == '.json':
        print(f"Saving results to JSON: {filepath}")

        # For JSON, optionally strip full SHAP values to reduce file size
        output_results = []
        for r in results:
            result_copy = r.copy()

            # Remove full_shap from explanations to reduce size (keep top_features)
            if 'stage1' in result_copy and result_copy['stage1']:
                if 'explanation' in result_copy['stage1']:
                    if 'full_shap' in result_copy['stage1']['explanation']:
                        del result_copy['stage1']['explanation']['full_shap']

            if 'stage2' in result_copy and result_copy['stage2']:
                if 'explanation' in result_copy['stage2']:
                    if 'full_shap' in result_copy['stage2']['explanation']:
                        del result_copy['stage2']['explanation']['full_shap']

            if 'stage3' in result_copy and result_copy['stage3']:
                if 'explanation' in result_copy['stage3']:
                    if 'full_shap' in result_copy['stage3']['explanation']:
                        del result_copy['stage3']['explanation']['full_shap']

            output_results.append(result_copy)

        with open(filepath, 'w') as f:
            json.dump(output_results, f, indent=2)

    elif filepath.suffix.lower() == '.csv':
        print(f"Saving results to CSV: {filepath}")
        # Flatten results for CSV format
        rows = []
        for r in results:
            row = {
                'sample_id': r['sample_id'],
                'original_id': r.get('original_id'),
                'stage1_prediction': r['stage1']['prediction'],
                'stage1_confidence': r['stage1']['confidence'],
                'stage1_de_probability': r['stage1']['de_probability'],
            }

            if r['stage2'] is not None:
                row['stage2_prediction'] = r['stage2']['prediction']
                row['stage2_confidence'] = r['stage2']['confidence']
                row['stage3_prediction'] = None
                row['stage3_confidence'] = None

            if r['stage3'] is not None:
                row['stage2_prediction'] = None
                row['stage2_confidence'] = None
                row['stage3_prediction'] = r['stage3']['prediction']
                row['stage3_prediction_full'] = r['stage3']['prediction_full_name']
                row['stage3_confidence'] = r['stage3']['confidence']
                # Add top 3 subtypes
                for i, item in enumerate(r['stage3'].get('top_3', [])[:3]):
                    row[f'stage3_subtype_{i+1}'] = item['subtype']
                    row[f'stage3_prob_{i+1}'] = item['probability']

            row['summary'] = r['summary']

            # Add top contributing features if SHAP was computed
            if 'top_contributing_features' in r:
                for i, feat in enumerate(r['top_contributing_features'][:5]):
                    row[f'top_feature_{i+1}'] = feat['feature']
                    row[f'top_feature_{i+1}_shap'] = feat['shap_value']
                    row[f'top_feature_{i+1}_direction'] = feat['direction']

            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)

    else:
        raise ValueError(f"Unsupported output format: {filepath.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch inference with Hierarchical Cognitive Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python batch_inference.py patients.csv results.json
    python batch_inference.py patients.json results.csv
    python batch_inference.py --help-features
        """
    )
    parser.add_argument('input', nargs='?', help='Input CSV or JSON file with patient data')
    parser.add_argument('output', nargs='?', help='Output file for results (JSON or CSV)')
    parser.add_argument('--help-features', action='store_true',
                        help='Print list of expected input features')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default=None,
                        help='Device to use for inference')
    parser.add_argument('--shap', action='store_true',
                        help='Compute SHAP explanations (slower, ~30-60s per sample)')
    parser.add_argument('--n-top-features', type=int, default=10,
                        help='Number of top SHAP features to return (default: 10)')

    args = parser.parse_args()

    # Initialize classifier
    print("=" * 70)
    print("Hierarchical Cognitive Impairment Classifier - Batch Inference")
    print("=" * 70)
    print()

    classifier = HierarchicalCognitiveClassifier(device=args.device)

    # Print features if requested
    if args.help_features:
        features = classifier.get_feature_list()
        print("\n" + "=" * 70)
        print(f"EXPECTED INPUT FEATURES ({len(features)} total)")
        print("=" * 70)
        print("\nFeatures should be columns in your CSV or keys in your JSON.")
        print("Missing values can be: empty, NaN, or NACC codes (-4, 88, 888, etc.)")
        print("\nFeature List:")
        for i, feat in enumerate(features):
            print(f"  {i+1:3d}. {feat}")
        return

    # Validate arguments
    if not args.input or not args.output:
        parser.print_help()
        print("\nError: Both input and output files are required.")
        sys.exit(1)

    # Load input data
    print()
    data = load_input_data(args.input)
    print(f"Loaded {len(data)} samples with {len(data.columns)} columns")

    # Check for ID column
    id_col = None
    for col in ['ID', 'id', 'patient_id', 'sample_id', 'NACCID']:
        if col in data.columns:
            id_col = col
            print(f"Using '{id_col}' as sample identifier")
            break

    # Run inference
    if args.shap:
        print(f"\nRunning inference with SHAP explanations on {len(data)} samples...")
        print("Note: SHAP computation is slow (~30-60 seconds per sample)")
    else:
        print(f"\nRunning inference on {len(data)} samples...")

    start_time = datetime.now()

    if args.shap:
        results = classifier.predict_with_explanations(
            data,
            n_top_features=args.n_top_features,
            compute_shap=True
        )
    else:
        results = classifier.predict(data)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"Inference completed in {elapsed:.2f} seconds ({elapsed/len(data)*1000:.1f} ms/sample)")

    # Add original IDs if available
    if id_col:
        for i, r in enumerate(results):
            r['original_id'] = str(data.iloc[i][id_col])

    # Save results
    print()
    save_results(results, args.output, data)

    # Print summary
    print("\n" + "=" * 70)
    print("INFERENCE SUMMARY")
    print("=" * 70)

    dementia_count = sum(1 for r in results if r['stage1']['prediction'] == 'Dementia')
    non_dementia_count = len(results) - dementia_count

    print(f"\nTotal Samples: {len(results)}")
    print(f"  Dementia: {dementia_count} ({dementia_count/len(results)*100:.1f}%)")
    print(f"  Non-Dementia: {non_dementia_count} ({non_dementia_count/len(results)*100:.1f}%)")

    if dementia_count > 0:
        print(f"\nDementia Subtype Distribution:")
        subtype_counts = {}
        for r in results:
            if r['stage3'] is not None:
                subtype = r['stage3']['prediction']
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1

        for subtype, count in sorted(subtype_counts.items(), key=lambda x: -x[1]):
            full_name = classifier.SUBTYPE_NAMES.get(subtype, subtype)
            print(f"    {subtype}: {count} ({count/dementia_count*100:.1f}%) - {full_name}")

    print(f"\nResults saved to: {args.output}")
    print("=" * 70)


if __name__ == '__main__':
    main()
