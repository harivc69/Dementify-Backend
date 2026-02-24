#!/usr/bin/env python3
"""
Hierarchical Cognitive Impairment Classifier - Frontend API

This module provides a clean, simple API for the frontend engineer to integrate
the hierarchical dementia classifier into their application.

Features:
    - Multiple input formats: CSV, JSON, dict, DataFrame
    - Multiple output formats: JSON, dict, DataFrame
    - Full hierarchical inference (Stage 1 -> Stage 2/3)
    - SHAP explanations for model interpretability
    - Batch processing support

Quick Start:
    from api import CognitiveClassifierAPI

    api = CognitiveClassifierAPI()
    result = api.predict(patient_data)
    result_with_shap = api.predict_with_explanations(patient_data)

Author: UIUC Research Team
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from hierarchical_classifier import HierarchicalCognitiveClassifier


class CognitiveClassifierAPI:
    """
    Production API for the Hierarchical Cognitive Impairment Classifier.

    This class provides a clean interface for frontend integration with:
    - Multiple input/output formats
    - Fast inference and SHAP explanations
    - Batch processing capabilities

    Example:
        >>> api = CognitiveClassifierAPI()
        >>>
        >>> # Single patient prediction
        >>> result = api.predict({'his_NACCAGE': 75, 'bat_NACCMMSE': 22})
        >>> print(result['summary'])
        >>>
        >>> # Batch prediction from file
        >>> results = api.predict('patients.csv', output_format='dataframe')
        >>>
        >>> # With SHAP explanations
        >>> result = api.predict_with_explanations(patient_data)
        >>> print(result['top_features'])
    """

    # Dementia subtype full names
    SUBTYPE_NAMES = {
        'AD': "Alzheimer's Disease",
        'LBD': 'Lewy Body Dementia',
        'VD': 'Vascular Dementia',
        'FTD': 'Frontotemporal Dementia',
        'PRD': 'Prion Disease (CJD)',
        'NPH': 'Normal Pressure Hydrocephalus',
        'SEF': 'Systemic/Environmental Factors',
        'PSY': 'Psychiatric Causes',
        'TBI': 'Traumatic Brain Injury',
        'ODE': 'Other Dementia Etiologies'
    }

    # Stage 2 class names
    COGNITIVE_STATUS_NAMES = {
        'NC': 'Normal Cognition',
        'MCI': 'Mild Cognitive Impairment',
        'IMCI': 'MCI with Functional Impairment'
    }

    def __init__(self, device: Optional[str] = None, verbose: bool = True):
        """
        Initialize the API.

        Args:
            device: 'cuda' or 'cpu'. Auto-detected if None.
            verbose: Whether to print loading messages.
        """
        if not verbose:
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()

        self.classifier = HierarchicalCognitiveClassifier(device=device)
        self.device = self.classifier.device

        if not verbose:
            sys.stdout = old_stdout

        self._feature_list = None

    @property
    def feature_list(self) -> List[str]:
        """Get the list of 187 input features."""
        if self._feature_list is None:
            self._feature_list = self.classifier.get_feature_list()
        return self._feature_list

    @property
    def num_features(self) -> int:
        """Get the number of input features (187)."""
        return len(self.feature_list)

    # ==================== INPUT HANDLING ====================

    def _load_input(self, data: Union[str, Dict, List[Dict], pd.DataFrame]) -> pd.DataFrame:
        """
        Convert any supported input format to DataFrame.

        Args:
            data: Input data in one of these formats:
                - str: Path to CSV or JSON file
                - dict: Single patient data
                - list of dicts: Multiple patients
                - DataFrame: Already formatted data

        Returns:
            pd.DataFrame with patient data
        """
        if isinstance(data, pd.DataFrame):
            return data

        if isinstance(data, dict):
            return pd.DataFrame([data])

        if isinstance(data, list):
            return pd.DataFrame(data)

        if isinstance(data, str):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {data}")

            if path.suffix.lower() == '.csv':
                return pd.read_csv(path)
            elif path.suffix.lower() == '.json':
                with open(path, 'r') as f:
                    json_data = json.load(f)
                if isinstance(json_data, list):
                    return pd.DataFrame(json_data)
                else:
                    return pd.DataFrame([json_data])
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        raise TypeError(f"Unsupported input type: {type(data)}")

    # ==================== OUTPUT FORMATTING ====================

    def _format_output(
        self,
        results: List[Dict],
        output_format: str = 'dict'
    ) -> Union[List[Dict], pd.DataFrame, str]:
        """
        Format results to the requested output format.

        Args:
            results: List of prediction dictionaries
            output_format: One of 'dict', 'json', 'dataframe', 'csv'

        Returns:
            Formatted results
        """
        if output_format == 'dict':
            if len(results) == 0:
                return []
            return results if len(results) > 1 else results[0]

        elif output_format == 'json':
            # For single result, return object not array
            if len(results) == 1:
                return json.dumps(results[0], indent=2)
            return json.dumps(results, indent=2)

        elif output_format in ('dataframe', 'csv'):
            rows = []
            for r in results:
                row = self._flatten_result(r)
                rows.append(row)

            df = pd.DataFrame(rows)

            if output_format == 'csv':
                return df.to_csv(index=False)
            return df

        else:
            raise ValueError(f"Unknown output_format: {output_format}")

    def _flatten_result(self, result: Dict, n_shap_features: int = 20) -> Dict:
        """Flatten nested result dict for DataFrame/CSV output."""
        flat = {
            'sample_id': result.get('sample_id'),
            'original_id': result.get('original_id'),

            # Stage 1
            'stage1_prediction': result['stage1']['prediction'],
            'stage1_confidence': result['stage1']['confidence'],
            'de_probability': result['stage1']['de_probability'],
        }

        # Stage 2 (Non-Dementia path)
        if result.get('stage2'):
            flat['stage2_prediction'] = result['stage2']['prediction']
            flat['stage2_full_name'] = result['stage2']['prediction_full_name']
            flat['stage2_confidence'] = result['stage2']['confidence']

            if 'probabilities' in result['stage2']:
                for cls, prob in result['stage2']['probabilities'].items():
                    flat[f'stage2_prob_{cls}'] = prob

        # Stage 3 (Dementia path)
        if result.get('stage3'):
            flat['stage3_prediction'] = result['stage3']['prediction']
            flat['stage3_full_name'] = result['stage3']['prediction_full_name']
            flat['stage3_confidence'] = result['stage3']['confidence']

            # Top 3 subtypes
            if 'top_3' in result['stage3']:
                for i, item in enumerate(result['stage3']['top_3'][:3]):
                    flat[f'stage3_top{i+1}_subtype'] = item['subtype']
                    flat[f'stage3_top{i+1}_prob'] = item['probability']

            # All probabilities
            if 'all_probabilities' in result['stage3']:
                for subtype, prob in result['stage3']['all_probabilities'].items():
                    flat[f'stage3_prob_{subtype}'] = prob

        flat['summary'] = result.get('summary', '')

        # SHAP features if present - include up to n_shap_features
        if 'top_contributing_features' in result:
            for i, feat in enumerate(result['top_contributing_features'][:n_shap_features]):
                flat[f'shap_feature_{i+1}'] = feat['feature']
                flat[f'shap_value_{i+1}'] = feat['shap_value']
                flat[f'shap_direction_{i+1}'] = feat['direction']

        return flat

    # ==================== MAIN API METHODS ====================

    def predict(
        self,
        data: Union[str, Dict, List[Dict], pd.DataFrame],
        output_format: str = 'dict'
    ) -> Union[Dict, List[Dict], pd.DataFrame, str]:
        """
        Run hierarchical prediction on patient data.

        This method runs the full 3-stage hierarchical pipeline:
        - Stage 1: Dementia vs Non-Dementia screening
        - Stage 2: NC/MCI/iMCI classification (if Non-Dementia)
        - Stage 3: 10 dementia subtype classification (if Dementia)

        Args:
            data: Patient data in one of these formats:
                - str: Path to CSV or JSON file
                - dict: Single patient {feature: value}
                - list of dicts: Multiple patients
                - DataFrame: Pandas DataFrame

            output_format: Output format:
                - 'dict': Python dict (default)
                - 'json': JSON string
                - 'dataframe': Pandas DataFrame
                - 'csv': CSV string

        Returns:
            Prediction results in requested format.

            For single patient (dict format):
            {
                'sample_id': 0,
                'stage1': {
                    'prediction': 'Dementia' or 'Non-Dementia',
                    'confidence': float,
                    'de_probability': float,
                    'probabilities': {...}
                },
                'stage2': {...} or None,  # If Non-Dementia
                'stage3': {...} or None,  # If Dementia
                'summary': str
            }

        Example:
            >>> api = CognitiveClassifierAPI()
            >>>
            >>> # Single patient
            >>> result = api.predict({'his_NACCAGE': 75, 'bat_NACCMMSE': 22})
            >>> print(result['summary'])
            >>>
            >>> # From CSV file
            >>> results = api.predict('patients.csv')
            >>>
            >>> # Get as DataFrame
            >>> df = api.predict('patients.json', output_format='dataframe')
        """
        df = self._load_input(data)
        results = self.classifier.predict(df, return_probabilities=True)

        # Add original_id if present
        if 'original_id' in df.columns or 'ID' in df.columns or 'id' in df.columns:
            id_col = 'original_id' if 'original_id' in df.columns else ('ID' if 'ID' in df.columns else 'id')
            for i, r in enumerate(results):
                r['original_id'] = str(df.iloc[i][id_col])

        return self._format_output(results, output_format)

    def predict_with_explanations(
        self,
        data: Union[str, Dict, List[Dict], pd.DataFrame],
        n_top_features: int = 20,
        output_format: str = 'dict'
    ) -> Union[Dict, List[Dict], pd.DataFrame, str]:
        """
        Run hierarchical prediction with SHAP explanations.

        This method provides interpretable predictions by computing SHAP values
        for each active stage, showing which features contributed most to the
        prediction.

        **Note:** SHAP computation is slow (~3-5 minutes per patient on GPU).
        For batch processing, use predict() for fast inference and this method
        for selected patients that need explanation.

        Args:
            data: Patient data (same formats as predict())
            n_top_features: Number of top SHAP features to return (default: 20).
                           Set to 187 to get all features ranked by importance.
            output_format: Output format ('dict', 'json', 'dataframe', 'csv')

        Returns:
            Prediction results with SHAP explanations:
            {
                'sample_id': 0,
                'stage1': {
                    'prediction': ...,
                    'explanation': {
                        'top_features': [
                            {'feature': 'his_NACCAGE', 'shap_value': 0.15,
                             'direction': 'increases', 'importance': 0.15},
                            ...
                        ]
                    }
                },
                'stage2': {..., 'explanation': {...}},  # If Non-Dementia
                'stage3': {..., 'explanation': {...}},  # If Dementia
                'top_contributing_features': [...],  # Aggregated across stages
                'summary': str
            }

        Example:
            >>> api = CognitiveClassifierAPI()
            >>> result = api.predict_with_explanations(patient_data)
            >>>
            >>> # Access top features
            >>> for feat in result['top_contributing_features'][:5]:
            ...     print(f"{feat['feature']}: {feat['shap_value']:.3f}")
        """
        df = self._load_input(data)
        results = self.classifier.predict_with_explanations(
            df,
            n_top_features=n_top_features,
            compute_shap=True
        )

        # Add original_id if present
        if 'original_id' in df.columns or 'ID' in df.columns or 'id' in df.columns:
            id_col = 'original_id' if 'original_id' in df.columns else ('ID' if 'ID' in df.columns else 'id')
            for i, r in enumerate(results):
                r['original_id'] = str(df.iloc[i][id_col])

        return self._format_output(results, output_format)

    def predict_stage1_only(
        self,
        data: Union[str, Dict, List[Dict], pd.DataFrame],
        output_format: str = 'dict'
    ) -> Union[Dict, List[Dict], pd.DataFrame, str]:
        """
        Run only Stage 1 (Dementia screening) without subsequent stages.

        Useful for quick dementia screening without full subtype classification.

        Args:
            data: Patient data (same formats as predict())
            output_format: Output format

        Returns:
            Stage 1 results only:
            {
                'sample_id': 0,
                'prediction': 'Dementia' or 'Non-Dementia',
                'confidence': float,
                'de_probability': float,
                'probabilities': {'Dementia': float, 'Non-Dementia': float}
            }
        """
        df = self._load_input(data)
        results = []

        for idx in range(len(df)):
            sample = df.iloc[[idx]]
            stage1_probs = self.classifier._run_stage_inference(
                self.classifier.stage1_model, sample
            )

            de_prob = stage1_probs.get('DE', 0.5)
            is_dementia = de_prob >= 0.5

            result = {
                'sample_id': idx,
                'prediction': 'Dementia' if is_dementia else 'Non-Dementia',
                'confidence': de_prob if is_dementia else (1 - de_prob),
                'de_probability': de_prob,
                'probabilities': {
                    'Non-Dementia': 1 - de_prob,
                    'Dementia': de_prob
                }
            }

            # Add original_id if present
            for id_col in ['original_id', 'ID', 'id']:
                if id_col in sample.columns:
                    result['original_id'] = str(sample[id_col].values[0])
                    break

            results.append(result)

        return self._format_output(results, output_format)

    # ==================== UTILITY METHODS ====================

    def get_feature_info(self) -> Dict[str, Dict]:
        """
        Get detailed information about all 187 input features.

        Returns:
            Dict mapping feature names to their info:
            {
                'his_NACCAGE': {'type': 'numerical', 'category': 'Demographics'},
                ...
            }
        """
        return self.classifier.get_feature_info()

    def get_sample_input(self) -> Dict:
        """
        Get a sample input dict with all features set to missing.

        Useful for understanding the expected input format.

        Returns:
            Dict with all 187 features set to None (missing)
        """
        return {feat: None for feat in self.feature_list}

    def validate_input(self, data: Union[Dict, pd.DataFrame]) -> Dict:
        """
        Validate input data and report any issues.

        Args:
            data: Patient data to validate

        Returns:
            Validation report:
            {
                'valid': bool,
                'total_features': int,
                'provided_features': int,
                'missing_features': int,
                'unknown_features': list,
                'warnings': list
            }
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data

        provided = set(df.columns)
        expected = set(self.feature_list)

        missing = expected - provided
        unknown = provided - expected - {'ID', 'id', 'original_id', 'sample_id'}

        # Count non-null values
        if len(df) > 0:
            non_null = df[list(provided & expected)].notna().sum(axis=1).mean()
        else:
            non_null = 0

        warnings_list = []
        if len(missing) > 0:
            warnings_list.append(f"{len(missing)} expected features missing (will use imputation)")
        if len(unknown) > 0:
            warnings_list.append(f"{len(unknown)} unknown features will be ignored: {list(unknown)[:5]}")

        return {
            'valid': True,  # Always valid due to imputation
            'total_features': len(expected),
            'provided_features': len(provided & expected),
            'missing_features': len(missing),
            'unknown_features': list(unknown),
            'avg_non_null_features': non_null,
            'warnings': warnings_list
        }

    def save_results(
        self,
        results: Union[Dict, List[Dict]],
        filepath: str,
        include_shap: bool = True
    ):
        """
        Save prediction results to a file.

        Args:
            results: Prediction results from predict() or predict_with_explanations()
            filepath: Output file path (supports .json and .csv)
            include_shap: Whether to include full SHAP values (default: True)
        """
        path = Path(filepath)

        if isinstance(results, dict):
            results = [results]

        if not include_shap:
            # Strip full SHAP data to reduce file size
            results = self._strip_full_shap(results)

        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(results, f, indent=2)

        elif path.suffix.lower() == '.csv':
            rows = [self._flatten_result(r) for r in results]
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)

        else:
            raise ValueError(f"Unsupported output format: {path.suffix}")

    def _strip_full_shap(self, results: List[Dict]) -> List[Dict]:
        """Remove full_shap from results to reduce size."""
        import copy
        stripped = copy.deepcopy(results)

        for r in stripped:
            for stage in ['stage1', 'stage2', 'stage3']:
                if r.get(stage) and 'explanation' in r[stage]:
                    if 'full_shap' in r[stage]['explanation']:
                        del r[stage]['explanation']['full_shap']

        return stripped


# ==================== CONVENIENCE FUNCTIONS ====================

def create_api(device: Optional[str] = None, verbose: bool = False) -> CognitiveClassifierAPI:
    """
    Factory function to create the API.

    Args:
        device: 'cuda' or 'cpu' (auto-detected if None)
        verbose: Whether to print loading messages

    Returns:
        CognitiveClassifierAPI instance
    """
    return CognitiveClassifierAPI(device=device, verbose=verbose)


def quick_predict(data: Union[str, Dict, List[Dict], pd.DataFrame]) -> Dict:
    """
    Quick prediction without explicitly creating an API instance.

    Note: This creates a new API instance each time, so it's slower for
    multiple predictions. For batch processing, create an API instance
    and reuse it.

    Args:
        data: Patient data

    Returns:
        Prediction results
    """
    api = CognitiveClassifierAPI(verbose=False)
    return api.predict(data)


# ==================== MAIN ====================

if __name__ == '__main__':
    print("="*70)
    print("Hierarchical Cognitive Impairment Classifier API")
    print("="*70)

    # Initialize API
    api = CognitiveClassifierAPI()

    print(f"\nModel loaded on device: {api.device}")
    print(f"Number of input features: {api.num_features}")

    # Show sample usage
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)

    print("""
# Basic prediction
api = CognitiveClassifierAPI()
result = api.predict({'his_NACCAGE': 75, 'bat_NACCMMSE': 22})
print(result['summary'])

# From file
results = api.predict('patients.csv')

# With SHAP explanations
result = api.predict_with_explanations(patient_data)

# Output formats
json_str = api.predict(data, output_format='json')
df = api.predict(data, output_format='dataframe')

# Save results
api.save_results(results, 'output.json')
""")
