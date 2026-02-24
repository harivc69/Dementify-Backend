#!/usr/bin/env python3
"""
Comprehensive Test Suite for Hierarchical Cognitive Classifier

This test suite validates all functionality before shipping to frontend:
- Multiple input formats (dict, list, DataFrame, CSV, JSON)
- Multiple output formats (dict, json, dataframe, csv)
- Edge cases and error handling
- Batch processing
- SHAP explanations
- API functionality

Run with: python tests/test_comprehensive.py
"""

import sys
import os
import json
import tempfile
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# Test counters
TESTS_PASSED = 0
TESTS_FAILED = 0
TESTS_TOTAL = 0


def test(name):
    """Decorator for test functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global TESTS_PASSED, TESTS_FAILED, TESTS_TOTAL
            TESTS_TOTAL += 1
            print(f"\n[TEST {TESTS_TOTAL}] {name}...", end=" ")
            try:
                func(*args, **kwargs)
                print("PASSED")
                TESTS_PASSED += 1
                return True
            except Exception as e:
                print(f"FAILED")
                print(f"         Error: {e}")
                TESTS_FAILED += 1
                return False
        return wrapper
    return decorator


def assert_true(condition, message="Assertion failed"):
    """Assert helper."""
    if not condition:
        raise AssertionError(message)


def assert_equal(a, b, message=None):
    """Assert equality helper."""
    if a != b:
        raise AssertionError(message or f"Expected {a} == {b}")


def assert_in(item, container, message=None):
    """Assert item in container."""
    if item not in container:
        raise AssertionError(message or f"Expected {item} in {container}")


# ============================================================================
# TESTS
# ============================================================================

class TestSuite:
    """Comprehensive test suite."""

    def __init__(self):
        print("="*70)
        print("HIERARCHICAL CLASSIFIER - COMPREHENSIVE TEST SUITE")
        print("="*70)
        print("\nLoading models (this may take a moment)...")

        from api import CognitiveClassifierAPI
        self.api = CognitiveClassifierAPI(verbose=False)

        # Create sample data
        self.sample_patient = self._create_sample_patient()
        self.sample_batch = self._create_sample_batch(5)

        print(f"Models loaded on: {self.api.device}")
        print(f"Number of features: {self.api.num_features}")

    def _create_sample_patient(self) -> dict:
        """Create a sample patient dict."""
        features = self.api.feature_list
        patient = {feat: None for feat in features}

        # Set some known values
        patient['his_NACCAGE'] = 75
        patient['his_SEX'] = 2
        patient['his_EDUC'] = 16
        patient['bat_NACCMMSE'] = 22
        patient['original_id'] = 'TEST_001'

        return patient

    def _create_sample_batch(self, n: int) -> list:
        """Create a batch of sample patients."""
        batch = []
        for i in range(n):
            patient = self._create_sample_patient()
            patient['his_NACCAGE'] = 60 + i * 5  # Vary age
            patient['bat_NACCMMSE'] = 30 - i * 3  # Vary MMSE
            patient['original_id'] = f'TEST_{i:03d}'
            batch.append(patient)
        return batch

    # ==================== INPUT FORMAT TESTS ====================

    @test("Input: Single dict")
    def test_input_dict(self):
        result = self.api.predict(self.sample_patient)
        assert_in('stage1', result)
        assert_in('prediction', result['stage1'])

    @test("Input: List of dicts")
    def test_input_list(self):
        results = self.api.predict(self.sample_batch)
        assert_true(isinstance(results, list), "Should return list for multiple inputs")
        assert_equal(len(results), 5)

    @test("Input: Pandas DataFrame")
    def test_input_dataframe(self):
        df = pd.DataFrame(self.sample_batch)
        results = self.api.predict(df)
        assert_true(isinstance(results, list))
        assert_equal(len(results), 5)

    @test("Input: CSV file")
    def test_input_csv(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df = pd.DataFrame(self.sample_batch)
            df.to_csv(f.name, index=False)

            results = self.api.predict(f.name)
            assert_equal(len(results), 5)

            os.unlink(f.name)

    @test("Input: JSON file")
    def test_input_json(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump(self.sample_batch, f)
            f.flush()

            results = self.api.predict(f.name)
            assert_equal(len(results), 5)

            os.unlink(f.name)

    @test("Input: JSON file (single patient)")
    def test_input_json_single(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            json.dump(self.sample_patient, f)
            f.flush()

            result = self.api.predict(f.name)
            assert_in('stage1', result)

            os.unlink(f.name)

    # ==================== OUTPUT FORMAT TESTS ====================

    @test("Output: dict format")
    def test_output_dict(self):
        result = self.api.predict(self.sample_patient, output_format='dict')
        assert_true(isinstance(result, dict))

    @test("Output: json format")
    def test_output_json(self):
        result = self.api.predict(self.sample_patient, output_format='json')
        assert_true(isinstance(result, str))
        parsed = json.loads(result)
        assert_in('stage1', parsed)

    @test("Output: dataframe format")
    def test_output_dataframe(self):
        result = self.api.predict(self.sample_batch, output_format='dataframe')
        assert_true(isinstance(result, pd.DataFrame))
        assert_equal(len(result), 5)

    @test("Output: csv format")
    def test_output_csv(self):
        result = self.api.predict(self.sample_batch, output_format='csv')
        assert_true(isinstance(result, str))
        assert_in('stage1_prediction', result)

    # ==================== PREDICTION RESULT TESTS ====================

    @test("Result: Contains all required fields")
    def test_result_fields(self):
        result = self.api.predict(self.sample_patient)
        assert_in('sample_id', result)
        assert_in('stage1', result)
        assert_in('summary', result)

        # Stage 1 fields
        assert_in('prediction', result['stage1'])
        assert_in('confidence', result['stage1'])
        assert_in('de_probability', result['stage1'])
        assert_in('probabilities', result['stage1'])

    @test("Result: Stage 1 prediction is valid")
    def test_stage1_prediction(self):
        result = self.api.predict(self.sample_patient)
        assert_in(result['stage1']['prediction'], ['Dementia', 'Non-Dementia'])
        assert_true(0 <= result['stage1']['confidence'] <= 1)
        assert_true(0 <= result['stage1']['de_probability'] <= 1)

    @test("Result: Stage 2 OR Stage 3 present (not both)")
    def test_stage2_or_stage3(self):
        result = self.api.predict(self.sample_patient)

        if result['stage1']['prediction'] == 'Dementia':
            assert_true(result['stage3'] is not None)
            assert_true(result['stage2'] is None)
        else:
            assert_true(result['stage2'] is not None)
            assert_true(result['stage3'] is None)

    @test("Result: Stage 2 fields (if Non-Dementia)")
    def test_stage2_fields(self):
        # Create patient likely to be Non-Dementia
        patient = self.sample_patient.copy()
        patient['bat_NACCMMSE'] = 28  # High MMSE

        result = self.api.predict(patient)

        if result['stage2']:
            assert_in('prediction', result['stage2'])
            assert_in('prediction_full_name', result['stage2'])
            assert_in('confidence', result['stage2'])
            assert_in(result['stage2']['prediction'], ['NC', 'MCI', 'IMCI'])

    @test("Result: Stage 3 fields (if Dementia)")
    def test_stage3_fields(self):
        # Create patient likely to be Dementia
        patient = self.sample_patient.copy()
        patient['bat_NACCMMSE'] = 10  # Low MMSE
        patient['his_NACCAGE'] = 85  # Older

        result = self.api.predict(patient)

        if result['stage3']:
            assert_in('prediction', result['stage3'])
            assert_in('prediction_full_name', result['stage3'])
            assert_in('confidence', result['stage3'])
            assert_in('top_3', result['stage3'])
            assert_equal(len(result['stage3']['top_3']), 3)

    @test("Result: Probabilities sum correctly")
    def test_probabilities(self):
        result = self.api.predict(self.sample_patient)

        # Stage 1 probabilities
        probs = result['stage1']['probabilities']
        total = probs['Dementia'] + probs['Non-Dementia']
        assert_true(0.99 <= total <= 1.01, f"Stage 1 probs should sum to 1, got {total}")

    @test("Result: original_id preserved")
    def test_original_id(self):
        patient = self.sample_patient.copy()
        patient['original_id'] = 'CUSTOM_ID_123'

        result = self.api.predict(patient)
        assert_equal(result.get('original_id'), 'CUSTOM_ID_123')

    # ==================== EDGE CASE TESTS ====================

    @test("Edge: All features missing (None)")
    def test_all_missing(self):
        patient = {feat: None for feat in self.api.feature_list}
        result = self.api.predict(patient)
        assert_in('stage1', result)
        assert_in('prediction', result['stage1'])

    @test("Edge: All features as -4 (NACC missing code)")
    def test_nacc_missing_codes(self):
        patient = {feat: -4 for feat in self.api.feature_list}
        result = self.api.predict(patient)
        assert_in('stage1', result)

    @test("Edge: Mixed missing codes")
    def test_mixed_missing(self):
        patient = {feat: None for feat in self.api.feature_list}
        patient['his_NACCAGE'] = 75
        patient['bat_NACCMMSE'] = 88  # Missing code
        patient['his_SEX'] = 9  # Missing code

        result = self.api.predict(patient)
        assert_in('stage1', result)

    @test("Edge: Only one feature provided")
    def test_minimal_features(self):
        patient = {'his_NACCAGE': 75}
        result = self.api.predict(patient)
        assert_in('stage1', result)

    @test("Edge: Extra unknown features ignored")
    def test_extra_features(self):
        patient = self.sample_patient.copy()
        patient['UNKNOWN_FEATURE_1'] = 123
        patient['UNKNOWN_FEATURE_2'] = 'abc'

        result = self.api.predict(patient)
        assert_in('stage1', result)

    @test("Edge: Empty batch")
    def test_empty_batch(self):
        df = pd.DataFrame(columns=self.api.feature_list)
        results = self.api.predict(df)
        assert_true(isinstance(results, list) or results == [])

    @test("Edge: Numeric string values")
    def test_string_numbers(self):
        patient = self.sample_patient.copy()
        patient['his_NACCAGE'] = '75'
        patient['bat_NACCMMSE'] = '22'

        result = self.api.predict(patient)
        # Result should be a dict for single patient
        if isinstance(result, list):
            result = result[0]
        assert_in('stage1', result)

    # ==================== BATCH PROCESSING TESTS ====================

    @test("Batch: 10 patients")
    def test_batch_10(self):
        batch = self._create_sample_batch(10)
        results = self.api.predict(batch)
        assert_equal(len(results), 10)

    @test("Batch: 50 patients")
    def test_batch_50(self):
        batch = self._create_sample_batch(50)
        start = time.time()
        results = self.api.predict(batch)
        elapsed = time.time() - start

        assert_equal(len(results), 50)
        print(f"({elapsed:.2f}s, {elapsed/50*1000:.0f}ms/sample)", end=" ")

    @test("Batch: Consistent results")
    def test_batch_consistency(self):
        # Same patient should give same result
        patient = self.sample_patient.copy()
        batch = [patient.copy() for _ in range(5)]

        results = self.api.predict(batch)

        # All should have same prediction
        predictions = [r['stage1']['prediction'] for r in results]
        assert_true(len(set(predictions)) == 1, "Same patient should get same prediction")

    # ==================== STAGE 1 ONLY TEST ====================

    @test("Stage 1 only prediction")
    def test_stage1_only(self):
        result = self.api.predict_stage1_only(self.sample_patient)
        assert_in('prediction', result)
        assert_in('de_probability', result)
        assert_true('stage2' not in result)
        assert_true('stage3' not in result)

    # ==================== UTILITY TESTS ====================

    @test("Utility: Feature list")
    def test_feature_list(self):
        features = self.api.feature_list
        assert_equal(len(features), 187)
        assert_in('his_NACCAGE', features)
        assert_in('bat_NACCMMSE', features)

    @test("Utility: Validate input")
    def test_validate_input(self):
        report = self.api.validate_input(self.sample_patient)
        assert_in('valid', report)
        assert_in('total_features', report)
        assert_equal(report['total_features'], 187)

    @test("Utility: Get sample input")
    def test_sample_input(self):
        sample = self.api.get_sample_input()
        assert_equal(len(sample), 187)
        assert_true(all(v is None for v in sample.values()))

    @test("Utility: Save results to JSON")
    def test_save_json(self):
        result = self.api.predict(self.sample_patient)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            self.api.save_results(result, f.name)

            with open(f.name, 'r') as rf:
                loaded = json.load(rf)

            assert_true(isinstance(loaded, list))
            os.unlink(f.name)

    @test("Utility: Save results to CSV")
    def test_save_csv(self):
        results = self.api.predict(self.sample_batch)

        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            self.api.save_results(results, f.name)

            df = pd.read_csv(f.name)
            assert_equal(len(df), 5)

            os.unlink(f.name)

    # ==================== ERROR HANDLING TESTS ====================

    @test("Error: Invalid file path")
    def test_invalid_file(self):
        try:
            self.api.predict('/nonexistent/path/file.csv')
            raise AssertionError("Should raise FileNotFoundError")
        except FileNotFoundError:
            pass  # Expected

    @test("Error: Invalid output format")
    def test_invalid_output_format(self):
        try:
            self.api.predict(self.sample_patient, output_format='invalid')
            raise AssertionError("Should raise ValueError")
        except ValueError:
            pass  # Expected

    @test("Error: Invalid file extension")
    def test_invalid_extension(self):
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
            try:
                self.api.predict(f.name)
                raise AssertionError("Should raise ValueError")
            except ValueError:
                pass  # Expected
            finally:
                os.unlink(f.name)

    # ==================== RUN ALL TESTS ====================

    def run_all(self):
        """Run all tests."""
        print("\n" + "="*70)
        print("RUNNING TESTS")
        print("="*70)

        # Input format tests
        self.test_input_dict()
        self.test_input_list()
        self.test_input_dataframe()
        self.test_input_csv()
        self.test_input_json()
        self.test_input_json_single()

        # Output format tests
        self.test_output_dict()
        self.test_output_json()
        self.test_output_dataframe()
        self.test_output_csv()

        # Result field tests
        self.test_result_fields()
        self.test_stage1_prediction()
        self.test_stage2_or_stage3()
        self.test_stage2_fields()
        self.test_stage3_fields()
        self.test_probabilities()
        self.test_original_id()

        # Edge case tests
        self.test_all_missing()
        self.test_nacc_missing_codes()
        self.test_mixed_missing()
        self.test_minimal_features()
        self.test_extra_features()
        self.test_empty_batch()
        self.test_string_numbers()

        # Batch tests
        self.test_batch_10()
        self.test_batch_50()
        self.test_batch_consistency()

        # Stage 1 only
        self.test_stage1_only()

        # Utility tests
        self.test_feature_list()
        self.test_validate_input()
        self.test_sample_input()
        self.test_save_json()
        self.test_save_csv()

        # Error handling tests
        self.test_invalid_file()
        self.test_invalid_output_format()
        self.test_invalid_extension()

        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"\nTotal:  {TESTS_TOTAL}")
        print(f"Passed: {TESTS_PASSED}")
        print(f"Failed: {TESTS_FAILED}")

        if TESTS_FAILED == 0:
            print("\n*** ALL TESTS PASSED ***")
        else:
            print(f"\n*** {TESTS_FAILED} TESTS FAILED ***")

        return TESTS_FAILED == 0


def run_shap_test():
    """Run SHAP test separately (slow)."""
    global TESTS_PASSED, TESTS_FAILED, TESTS_TOTAL

    print("\n" + "="*70)
    print("SHAP EXPLANATION TEST (this will take ~5 minutes)")
    print("="*70)

    from api import CognitiveClassifierAPI
    api = CognitiveClassifierAPI(verbose=False)

    features = api.feature_list
    patient = {feat: None for feat in features}
    patient['his_NACCAGE'] = 75
    patient['bat_NACCMMSE'] = 22

    TESTS_TOTAL += 1
    print(f"\n[TEST {TESTS_TOTAL}] SHAP explanations...", end=" ")

    try:
        start = time.time()
        result = api.predict_with_explanations(patient, n_top_features=10)
        elapsed = time.time() - start

        # Verify SHAP results
        assert_in('top_contributing_features', result)
        assert_true(len(result['top_contributing_features']) > 0)

        # Verify stage explanations
        assert_in('explanation', result['stage1'])
        assert_in('top_features', result['stage1']['explanation'])

        print(f"PASSED ({elapsed:.1f}s)")
        print(f"         Top feature: {result['top_contributing_features'][0]['feature']}")
        TESTS_PASSED += 1
        return True

    except Exception as e:
        print(f"FAILED")
        print(f"         Error: {e}")
        TESTS_FAILED += 1
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run comprehensive tests')
    parser.add_argument('--shap', action='store_true', help='Include SHAP test (slow)')
    parser.add_argument('--quick', action='store_true', help='Quick test only')
    args = parser.parse_args()

    suite = TestSuite()

    if args.quick:
        # Just run a few basic tests
        suite.test_input_dict()
        suite.test_output_dict()
        suite.test_result_fields()
        print("\nQuick test completed.")
    else:
        success = suite.run_all()

        if args.shap:
            run_shap_test()

        print("\n" + "="*70)
        print(f"FINAL: {TESTS_PASSED}/{TESTS_TOTAL} tests passed")
        print("="*70)

        sys.exit(0 if TESTS_FAILED == 0 else 1)
