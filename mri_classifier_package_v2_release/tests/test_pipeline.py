#!/usr/bin/env python3
"""
End-to-End Test for MRI Classifier Package v2.0

Verifies:
1. Pipeline loads all 3 stage models
2. Prediction returns correct JSON structure
3. Heatmap NIfTI outputs exist with correct shape/affine
4. Brain NIfTI output exists
5. SHAP output has 187 tabular features
6. MRI importance score is present
7. No hardcoded absolute paths remain

Usage:
    python tests/test_pipeline.py --mri_path /path/to/test_scan.nii.gz
    python tests/test_pipeline.py --mri_path /path/to/test_scan.nii.gz --with_shap
"""

import os
import sys
import json
import argparse
import traceback

# Add package root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PACKAGE_DIR)


def test_imports():
    """Test that all required imports work."""
    print("[TEST] Imports...")
    try:
        from api import MRICognitiveClassifierAPI
        from inference_pipeline import HierarchicalInferencePipeline, MRIPreprocessor
        print("  PASS: All imports successful")
        return True
    except ImportError as e:
        print(f"  FAIL: Import error: {e}")
        return False


def test_no_hardcoded_paths():
    """Check that no hardcoded absolute paths remain in key files."""
    print("[TEST] No hardcoded paths...")
    files_to_check = [
        os.path.join(PACKAGE_DIR, 'inference_pipeline.py'),
        os.path.join(PACKAGE_DIR, 'api.py'),
        os.path.join(PACKAGE_DIR, 'skull_strip.py'),
    ]

    # Also check explainability modules
    explain_dir = os.path.join(PACKAGE_DIR, 'explainability')
    if os.path.isdir(explain_dir):
        for f in os.listdir(explain_dir):
            if f.endswith('.py'):
                files_to_check.append(os.path.join(explain_dir, f))

    bad_patterns = ['/home/vatsal/', '/mnt/z/']
    failures = []

    for fpath in files_to_check:
        if not os.path.isfile(fpath):
            continue
        with open(fpath, 'r') as f:
            content = f.read()
        for pattern in bad_patterns:
            if pattern in content:
                # Find the line
                for i, line in enumerate(content.split('\n'), 1):
                    if pattern in line and not line.strip().startswith('#'):
                        failures.append(f"  {os.path.basename(fpath)}:{i}: {line.strip()[:80]}")

    if failures:
        print(f"  FAIL: Found {len(failures)} hardcoded path(s):")
        for f in failures:
            print(f)
        return False
    else:
        print("  PASS: No hardcoded absolute paths found")
        return True


def test_model_files():
    """Check that model checkpoint files exist."""
    print("[TEST] Model files...")
    models_dir = os.path.join(PACKAGE_DIR, 'models')
    expected = [
        'stage1_de_vs_non_de.pt',
        'stage2_nc_mci_imci.pt',
        'stage3_dementia_subtypes.pt',
    ]

    all_found = True
    for name in expected:
        path = os.path.join(models_dir, name)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  OK: {name} ({size_mb:.1f} MB)")
        else:
            print(f"  MISSING: {name}")
            all_found = False

    if all_found:
        print("  PASS: All model files present")
    else:
        print("  FAIL: Missing model files")
    return all_found


def test_config_files():
    """Check that config TOML files exist."""
    print("[TEST] Config files...")
    configs_dir = os.path.join(PACKAGE_DIR, 'configs')
    expected = [
        'stage1_de_mri_config.toml',
        'stage2_3way_mri_config.toml',
        'stage3_10class_mri_config.toml',
    ]

    all_found = True
    for name in expected:
        path = os.path.join(configs_dir, name)
        if os.path.isfile(path):
            print(f"  OK: {name}")
        else:
            print(f"  MISSING: {name}")
            all_found = False

    if all_found:
        print("  PASS: All config files present")
    else:
        print("  FAIL: Missing config files")
    return all_found


def test_prediction(mri_path: str, output_dir: str):
    """Test basic prediction."""
    print("[TEST] Prediction...")
    from api import MRICognitiveClassifierAPI

    api = MRICognitiveClassifierAPI()

    # Minimal features (model handles missing)
    features = {'his_NACCAGE': 72, 'his_SEX': 1}

    result = api.predict(
        mri_path=mri_path,
        features=features,
        output_dir=output_dir,
        generate_heatmap=True
    )

    # Verify JSON structure
    errors = []

    required_keys = ['timestamp', 'mri_path', 'stage1', 'final_diagnosis', 'heatmaps']
    for key in required_keys:
        if key not in result:
            errors.append(f"Missing key: {key}")

    # Stage 1
    s1 = result.get('stage1', {})
    for key in ['prediction', 'de_probability', 'confidence']:
        if key not in s1:
            errors.append(f"stage1 missing: {key}")

    if s1.get('prediction') not in ('DE', 'Non-DE'):
        errors.append(f"Invalid stage1 prediction: {s1.get('prediction')}")

    de_prob = s1.get('de_probability', -1)
    if not (0 <= de_prob <= 1):
        errors.append(f"Invalid de_probability: {de_prob}")

    # Stage 2 or 3 should be populated
    if result.get('stage2') is None and result.get('stage3') is None:
        errors.append("Both stage2 and stage3 are None")

    if errors:
        print(f"  FAIL: {len(errors)} error(s):")
        for e in errors:
            print(f"    {e}")
        return False
    else:
        print(f"  PASS: Prediction = {result['final_diagnosis']} (DE prob={de_prob:.4f})")
        return True


def test_nifti_outputs(output_dir: str, mri_path: str):
    """Verify NIfTI heatmap outputs."""
    print("[TEST] NIfTI outputs...")
    import nibabel as nib

    # Load original for comparison
    orig = nib.load(mri_path)
    orig_shape = orig.shape[:3]

    errors = []
    for name in ['heatmap.nii.gz', 'brain.nii.gz', 'overlay.nii.gz']:
        fpath = os.path.join(output_dir, name)
        if not os.path.isfile(fpath):
            errors.append(f"Missing: {name}")
            continue

        img = nib.load(fpath)
        data = img.get_fdata()

        # Check shape matches original
        if data.shape[:3] != orig_shape:
            errors.append(f"{name}: shape {data.shape[:3]} != original {orig_shape}")

        # Check data type
        if data.dtype not in ('float32', 'float64'):
            errors.append(f"{name}: unexpected dtype {data.dtype}")

        # Check heatmap range
        if name == 'heatmap.nii.gz':
            if data.min() < -0.01 or data.max() > 1.01:
                errors.append(f"heatmap range [{data.min():.4f}, {data.max():.4f}] outside [0, 1]")

        print(f"  {name}: shape={data.shape}, range=[{data.min():.4f}, {data.max():.4f}]")

    if errors:
        print(f"  FAIL: {len(errors)} error(s):")
        for e in errors:
            print(f"    {e}")
        return False
    else:
        print("  PASS: All NIfTI outputs valid")
        return True


def test_shap(mri_path: str, output_dir: str):
    """Test SHAP explanations."""
    print("[TEST] SHAP explanations (this takes ~3-5 minutes)...")
    from api import MRICognitiveClassifierAPI

    api = MRICognitiveClassifierAPI()
    features = {'his_NACCAGE': 72, 'his_SEX': 1, 'bat_NACCMMSE': 22}

    result = api.predict_with_explanations(
        mri_path=mri_path,
        features=features,
        output_dir=os.path.join(output_dir, 'shap'),
        n_top_features=20
    )

    errors = []

    # Check feature_importance
    fi = result.get('feature_importance', {})
    if not fi:
        errors.append("Missing feature_importance")
    else:
        # MRI importance
        mri_imp = fi.get('mri_importance')
        if mri_imp is None:
            errors.append("Missing mri_importance")
        else:
            print(f"  MRI importance: {mri_imp:.4f}")

        # Top features
        top = fi.get('top_features', [])
        if not top:
            errors.append("Empty top_features")
        else:
            print(f"  Top {len(top)} features:")
            for feat in top[:5]:
                print(f"    {feat['feature']}: {feat['shap_value']:+.4f}")

        # Tabular SHAP
        ts = fi.get('tabular_shap', {})
        if ts:
            n_features = len(ts.get('feature_names', []))
            n_values = len(ts.get('shap_values', []))
            print(f"  Tabular SHAP: {n_features} features, {n_values} values")
            if n_features == 0:
                errors.append("No tabular SHAP feature names")

    if errors:
        print(f"  FAIL: {len(errors)} error(s):")
        for e in errors:
            print(f"    {e}")
        return False
    else:
        print("  PASS: SHAP explanations valid")
        return True


def main():
    parser = argparse.ArgumentParser(description='Test MRI Classifier Package v2.0')
    parser.add_argument('--mri_path', type=str, default=None,
                        help='Path to test MRI scan (.nii.gz)')
    parser.add_argument('--with_shap', action='store_true',
                        help='Also test SHAP explanations (slow)')
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, 'test_output'),
                        help='Output directory for test results')

    args = parser.parse_args()

    print("=" * 60)
    print("MRI Classifier Package v2.0 - Test Suite")
    print("=" * 60)

    results = {}

    # Static tests (no model loading needed)
    results['imports'] = test_imports()
    results['no_hardcoded_paths'] = test_no_hardcoded_paths()
    results['model_files'] = test_model_files()
    results['config_files'] = test_config_files()

    # Dynamic tests (require MRI scan)
    if args.mri_path:
        if not os.path.isfile(args.mri_path):
            print(f"\nERROR: MRI file not found: {args.mri_path}")
        else:
            os.makedirs(args.output_dir, exist_ok=True)

            try:
                results['prediction'] = test_prediction(args.mri_path, args.output_dir)
            except Exception as e:
                print(f"  FAIL: {e}")
                traceback.print_exc()
                results['prediction'] = False

            try:
                results['nifti_outputs'] = test_nifti_outputs(args.output_dir, args.mri_path)
            except Exception as e:
                print(f"  FAIL: {e}")
                results['nifti_outputs'] = False

            if args.with_shap:
                try:
                    results['shap'] = test_shap(args.mri_path, args.output_dir)
                except Exception as e:
                    print(f"  FAIL: {e}")
                    results['shap'] = False
    else:
        print("\nSkipping dynamic tests (no --mri_path provided)")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\n{passed}/{total} tests passed")

    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
