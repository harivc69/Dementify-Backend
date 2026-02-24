import io
import pandas as pd
import json
from typing import List, Dict, Union
import numpy as np
from app.core.exceptions import ValidationError

# NACC to Model Mapping

NACC_TO_MODEL = {
    # Demographics & History (his_)
    "BIRTHMO": "his_BIRTHMO", "BIRTHYR": "his_BIRTHYR", "SEX": "his_SEX",
    "HISPANIC": "his_HISPANIC", "HISPOR": "his_HISPOR", "RACE": "his_RACE",
    "RACESEC": "his_RACESEC", "RACETER": "his_RACETER", "PRIMLANG": "his_PRIMLANG",
    "EDUC": "his_EDUC", "MARISTAT": "his_MARISTAT", "NACCLIVS": "his_NACCLIVS",
    "INDEPEND": "his_INDEPEND", "RESIDENC": "his_RESIDENC", "HANDED": "his_HANDED",
    "NACCAGE": "his_NACCAGE", "NACCFAM": "his_NACCFAM", "NACCMOM": "his_NACCMOM",
    "NACCDAD": "his_NACCDAD", "NACCAM": "his_NACCAM", "NACCAMS": "his_NACCAMS",
    "NACCFM": "his_NACCFM", "NACCFMS": "his_NACCFMS", "NACCOM": "his_NACCOM",
    "NACCOMS": "his_NACCOMS", "NACCFADM": "his_NACCFADM", "NACCFFTD": "his_NACCFFTD",
    "TOBAC30": "his_TOBAC30", "TOBAC100": "his_TOBAC100", "SMOKYRS": "his_SMOKYRS",
    "PACKSPER": "his_PACKSPER", "QUITSMOK": "his_QUITSMOK", "ALCOCCAS": "his_ALCOCCAS",
    "ALCFREQ": "his_ALCFREQ", "CVHATT": "his_CVHATT", "HATTMULT": "his_HATTMULT",
    "HATTYEAR": "his_HATTYEAR", "CVAFIB": "his_CVAFIB", "CVANGIO": "his_CVANGIO",
    "CVBYPASS": "his_CVBYPASS", "CVPACDEF": "his_CVPACDEF", "CVPACE": "his_CVPACE",
    "CVCHF": "his_CVCHF", "CVANGINA": "his_CVANGINA", "CVHVALVE": "his_CVHVALVE",
    "CVOTHR": "his_CVOTHR", "CBSTROKE": "his_CBSTROKE", "STROKMUL": "his_STROKMUL",
    "NACCSTYR": "his_NACCSTYR", "CBTIA": "his_CBTIA", "TIAMULT": "his_TIAMULT",
    "NACCTIYR": "his_NACCTIYR", "PD": "his_PD", "PDYR": "his_PDYR",
    "PDOTHR": "his_PDOTHR", "PDOTHRYR": "his_PDOTHRYR", "SEIZURES": "his_SEIZURES",
    "NACCTBI": "his_NACCTBI", "TBI": "his_TBI", "TBIBRIEF": "his_TBIBRIEF",
    "TBIEXTEN": "his_TBIEXTEN", "TBIWOLOS": "his_TBIWOLOS", "TBIYEAR": "his_TBIYEAR",
    "NCOTHR": "his_NCOTHR", "DIABETES": "his_DIABETES", "DIABTYPE": "his_DIABTYPE",
    "HYPERTEN": "his_HYPERTEN", "HYPERCHO": "his_HYPERCHO", "B12DEF": "his_B12DEF",
    "THYROID": "his_THYROID", "ARTHRIT": "his_ARTHRIT", "ARTHTYPE": "his_ARTHTYPE",
    "ARTHUPEX": "his_ARTHUPEX", "ARTHLOEX": "his_ARTHLOEX", "ARTHSPIN": "his_ARTHSPIN",
    "ARTHUNK": "his_ARTHUNK", "INCONTU": "his_INCONTU", "INCONTF": "his_INCONTF",
    "APNEA": "his_APNEA", "RBD": "his_RBD", "INSOMN": "his_INSOMN",
    "OTHSLEEP": "his_OTHSLEEP", "ALCOHOL": "his_ALCOHOL", "ABUSOTHR": "his_ABUSOTHR",
    "PTSD": "his_PTSD", "BIPOLAR": "his_BIPOLAR", "SCHIZ": "his_SCHIZ",
    "DEP2YRS": "his_DEP2YRS", "DEPOTHR": "his_DEPOTHR", "ANXIETY": "his_ANXIETY",
    "OCD": "his_OCD", "NPSYDEV": "his_NPSYDEV", "PSYCDIS": "his_PSYCDIS",

    # Physical Examination (ph_)
    "HEIGHT": "ph_HEIGHT", "WEIGHT": "ph_WEIGHT", "BPSYS": "ph_BPSYS",
    "BPDIAS": "ph_BPDIAS", "HRATE": "ph_HRATE", "VISION": "ph_VISION",
    "VISCORR": "ph_VISCORR", "VISWCORR": "ph_VISWCORR", "HEARING": "ph_HEARING",
    "HEARAID": "ph_HEARAID", "HEARWAID": "ph_HEARWAID", "NACCBMI": "ph_NACCBMI",

    # Cognitive Battery (bat_)
    "NACCMMSE": "bat_NACCMMSE", "MOCATOTS": "bat_MOCATOTS", "NACCMOCA": "bat_NACCMOCA",
    "MOCBTOTS": "bat_MOCBTOTS", "NACCMOCB": "bat_NACCMOCB",

    # Medications (med_)
    "ANYMEDS": "med_ANYMEDS", "NACCAAAS": "med_NACCAAAS", "NACCAANX": "med_NACCAANX",
    "NACCAC": "med_NACCAC", "NACCACEI": "med_NACCACEI", "NACCADEP": "med_NACCADEP",
    "NACCADMD": "med_NACCADMD", "NACCAHTN": "med_NACCAHTN", "NACCANGI": "med_NACCANGI",
    "NACCAPSY": "med_NACCAPSY", "NACCBETA": "med_NACCBETA", "NACCCCBS": "med_NACCCCBS",
    "NACCDBMD": "med_NACCDBMD", "NACCDIUR": "med_NACCDIUR", "NACCEMD": "med_NACCEMD",
    "NACCEPMD": "med_NACCEPMD", "NACCHTNC": "med_NACCHTNC", "NACCLIPL": "med_NACCLIPL",
    "NACCNSD": "med_NACCNSD", "NACCPDMD": "med_NACCPDMD", "NACCVASD": "med_NACCVASD",

    # Neurological Examination (exam_)
    "ABRUPT": "exam_ABRUPT", "STEPWISE": "exam_STEPWISE", "SOMATIC": "exam_SOMATIC",
    "EMOT": "exam_EMOT", "HXHYPER": "exam_HXHYPER", "HXSTROKE": "exam_HXSTROKE",
    "FOCLSYM": "exam_FOCLSYM", "FOCLSIGN": "exam_FOCLSIGN", "HACHIN": "exam_HACHIN",
    "CVDCOG": "exam_CVDCOG", "STROKCOG": "exam_STROKCOG", "NACCNREX": "exam_NACCNREX",
    "NORMEXAM": "exam_NORMEXAM", "FOCLDEF": "exam_FOCLDEF", "GAITDIS": "exam_GAITDIS",
    "EYEMOVE": "exam_EYEMOVE", "PARKSIGN": "exam_PARKSIGN", "RESTTRL": "exam_RESTTRL",
    "RESTTRR": "exam_RESTTRR", "SLOWINGL": "exam_SLOWINGL", "SLOWINGR": "exam_SLOWINGR",
    "RIGIDL": "exam_RIGIDL", "RIGIDR": "exam_RIGIDR", "BRADY": "exam_BRADY",
    "PARKGAIT": "exam_PARKGAIT", "POSTINST": "exam_POSTINST", "CVDSIGNS": "exam_CVDSIGNS",
    "CORTDEF": "exam_CORTDEF", "SIVDFIND": "exam_SIVDFIND", "CVDMOTL": "exam_CVDMOTL",
    "CVDMOTR": "exam_CVDMOTR", "CORTVISL": "exam_CORTVISL", "CORTVISR": "exam_CORTVISR",
    "SOMATL": "exam_SOMATL", "SOMATR": "exam_SOMATR", "POSTCORT": "exam_POSTCORT",
    "PSPCBS": "exam_PSPCBS", "EYEPSP": "exam_EYEPSP", "DYSPSP": "exam_DYSPSP",
    "AXIALPSP": "exam_AXIALPSP", "GAITPSP": "exam_GAITPSP", "APRAXSP": "exam_APRAXSP",
    "APRAXL": "exam_APRAXL", "APRAXR": "exam_APRAXR", "CORTSENL": "exam_CORTSENL",
    "CORTSENR": "exam_CORTSENR", "ATAXL": "exam_ATAXL", "ATAXR": "exam_ATAXR",
    "ALIENLML": "exam_ALIENLML", "ALIENLMR": "exam_ALIENLMR", "DYSTONL": "exam_DYSTONL",
    "DYSTONR": "exam_DYSTONR", "MYOCLLT": "exam_MYOCLLT", "MYOCLRT": "exam_MYOCLRT",
    "ALSFIND": "exam_ALSFIND", "GAITNPH": "exam_GAITNPH",
}




MISSING_CODES = {-4, 8, 9, 88, 99, 888, 999, 8888, 9999}

class DataService:
    @staticmethod
    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize input DataFrame:
        1. Remove duplicate columns.
        2. Rename columns from NACC format to model format.
        3. Convert missing value codes to NaN.
        """
        # Remove duplicate columns (keep first)
        df = df.loc[:, ~df.columns.duplicated()]

        # Rename columns
        rename_map = {}
        for col in df.columns:
            if col in NACC_TO_MODEL:
                rename_map[col] = NACC_TO_MODEL[col]
        
        if rename_map:
            df = df.rename(columns=rename_map)

        # Convert missing value codes to NaN
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                mask = df[col].isin(MISSING_CODES)
                if mask.any():
                    # Handle post-rename duplicates (if any)
                    # Even after deduplicating original names, renaming BIRTHMO to his_BIRTHMO 
                    # when his_BIRTHMO already exists would create a duplicate.
                    # So we deduplicate again after renaming if needed.
                    pass
        
        # Deduplicate again in case renames created new duplicates
        df = df.loc[:, ~df.columns.duplicated()]

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                mask = df[col].isin(MISSING_CODES)
                if mask.any():
                    df.loc[mask, col] = np.nan
                    
        return df





    @staticmethod
    def parse_csv(content: bytes) -> pd.DataFrame:
        """Parse CSV bytes into a pandas DataFrame."""
        try:
            decoded = content.decode('utf-8-sig')
            df = pd.read_csv(io.StringIO(decoded))
        except Exception as e:
            raise ValidationError(
                message="Failed to parse CSV file",
                code="FILE_2002",
                detail=str(e)
            )
            
        df = DataService.normalize_data(df)

        return df

    @staticmethod
    def parse_json(content: bytes) -> pd.DataFrame:
        """Parse JSON bytes into a pandas DataFrame."""
        try:
            data = json.loads(content.decode('utf-8'))
        except Exception as e:
            raise ValidationError(
                message="Failed to parse JSON file",
                code="VAL_1004",
                detail=str(e)
            )

        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        df = DataService.normalize_data(df)

        return df


    @staticmethod
    def prepare_manual_features(features: Dict, all_expected_features: List[str]) -> pd.DataFrame:
        """Prepare a single row DataFrame from manual feature dictionary."""
        input_data = {}
        # Pre-normalize input keys if they match NACC names
        normalized_features = {}
        for k, v in features.items():
            new_k = NACC_TO_MODEL.get(k, k)
            normalized_features[new_k] = v
            
        for feat in all_expected_features:
            if feat in normalized_features:
                input_data[feat] = normalized_features[feat]
            else:
                input_data[feat] = -4  # Missing value code for manual entry default
        
        # We don't run full normalize_data here because we manually set -4 defaults
        # But if the user provided specific missing codes, should we convert them?
        # For manual entry, assuming inputs are 'clean' or raw values. 
        # But let's respect the method's purpose: creating a strict DF for the model.
        return pd.DataFrame([input_data])
