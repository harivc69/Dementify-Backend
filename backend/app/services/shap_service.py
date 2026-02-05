import pandas as pd
from typing import List, Dict
from app.services.classifier_service import ClassifierService

class ShapService:
    def __init__(self, classifier_service: ClassifierService):
        self.classifier_service = classifier_service

    def get_explanations(self, df: pd.DataFrame, n_top_features: int = 20) -> List[Dict]:
        """Compute SHAP explanations for the given DataFrame."""
        results = self.classifier_service.classifier.predict_with_explanations(
            df,
            n_top_features=n_top_features,
            compute_shap=True
        )
        
        # Add original_id if present in input df
        id_columns = ['original_id', 'ID', 'id', 'patient_id', 'PatientID']
        for id_col in id_columns:
            if id_col in df.columns:
                for i, r in enumerate(results):
                    r['original_id'] = str(df.iloc[i][id_col])
                break
        
        return results
