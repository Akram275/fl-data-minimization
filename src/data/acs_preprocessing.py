"""
Preprocessing of the folktables ACS tasks into model inputs: missing-value imputation, one-hot encoding with
global category lists (identical columns for every state), standardized numerical features, and the
high-cardinality codes (occupation, place of birth) dropped. Used by src/data/acs_tabular.py.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional


# =============================================================================
# Feature Type Definitions
# =============================================================================

# Numerical features (continuous values that should be scaled)
NUMERICAL_FEATURES = {
    'AGEP',      # Age
    'WKHP',      # Work hours per week
    'PINCP',     # Total person's income
    'JWMNP',     # Travel time to work (minutes)
    'POVPIP',    # Income-to-poverty ratio
}

# High cardinality features (drop or use embedding - here we drop for MLP)
HIGH_CARDINALITY_FEATURES = {
    'OCCP',      # Occupation code (~500 categories)
    'POBP',      # Place of birth (~200 categories)
    'PUMA',      # Public use microdata area
    'POWPUMA',   # Place of work PUMA
    'ST',        # State code (50+ categories)
}

# Low cardinality categorical features (one-hot encode)
CATEGORICAL_FEATURES = {
    'COW',       # Class of worker
    'SCHL',      # Educational attainment
    'MAR',       # Marital status
    'RELP',      # Relationship to householder
    'SEX',       # Sex
    'RAC1P',     # Race
    'DIS',       # Disability status
    'ESP',       # Employment status of parents
    'CIT',       # Citizenship status
    'MIG',       # Mobility status (lived here 1 year ago)
    'MIL',       # Military service
    'ANC',       # Ancestry
    'NATIVITY',  # Nativity
    'DEAR',      # Hearing difficulty
    'DEYE',      # Vision difficulty
    'DREM',      # Cognitive difficulty
    'ESR',       # Employment status recode
    'FER',       # Gave birth in past 12 months
    'GCL',       # Grandparents living with grandchildren
    'JWTR',      # Means of transportation to work
}


# =============================================================================
# Global Category Definitions
# =============================================================================

# Define all possible categories for each categorical feature
# This ensures consistent one-hot encoding across all states/datasets

GLOBAL_CATEGORIES = {
    'COW': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],  # Class of worker
    'SCHL': [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0
    ],  # Educational attainment
    'MAR': [1.0, 2.0, 3.0, 4.0, 5.0],  # Marital status
    'RELP': [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0
    ],  # Relationship
    'SEX': [1.0, 2.0],  # Sex
    'RAC1P': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],  # Race
    'DIS': [1.0, 2.0],  # Disability status (1=with, 2=without)
    'ESP': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # Employment status of parents
    'CIT': [1.0, 2.0, 3.0, 4.0, 5.0],  # Citizenship
    'MIG': [1.0, 2.0, 3.0],  # Mobility status
    'MIL': [1.0, 2.0, 3.0, 4.0],  # Military service
    'ANC': [1.0, 2.0, 3.0, 4.0, 8.0],  # Ancestry
    'NATIVITY': [1.0, 2.0],  # Nativity (1=native, 2=foreign)
    'DEAR': [1.0, 2.0],  # Hearing difficulty
    'DEYE': [1.0, 2.0],  # Vision difficulty
    'DREM': [1.0, 2.0],  # Cognitive difficulty
    'ESR': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],  # Employment status recode
    'FER': [1.0, 2.0],  # Gave birth (1=yes, 2=no)
    'GCL': [1.0, 2.0],  # Grandparents with grandchildren
    'JWTR': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],  # Transportation to work
}


# =============================================================================
# Preprocessing Functions
# =============================================================================

def get_feature_types(df: pd.DataFrame) -> tuple:
    """
    Identify which features in the dataframe are numerical, categorical, or high-cardinality.

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (numerical_cols, categorical_cols, high_card_cols)
    """
    cols = set(df.columns)

    numerical_cols = list(cols & NUMERICAL_FEATURES)
    categorical_cols = list(cols & CATEGORICAL_FEATURES)
    high_card_cols = list(cols & HIGH_CARDINALITY_FEATURES)

    # Identify any remaining columns that weren't classified
    classified = set(numerical_cols) | set(categorical_cols) | set(high_card_cols)
    unclassified = cols - classified

    if unclassified:
        # Try to auto-classify based on dtype and unique values
        for col in unclassified:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                n_unique = df[col].nunique()
                if n_unique > 20:
                    # Likely numerical or high-cardinality
                    if n_unique > 100:
                        high_card_cols.append(col)
                    else:
                        numerical_cols.append(col)
                else:
                    categorical_cols.append(col)

    return numerical_cols, categorical_cols, high_card_cols


def impute_missing_values(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    num_strategy: str = 'median',
    cat_placeholder: float = -1.0
) -> pd.DataFrame:
    """
    Impute missing values in the DataFrame.

    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        num_strategy: Strategy for numerical imputation ('median', 'mean', 'zero')
        cat_placeholder: Placeholder value for missing categoricals

    Returns:
        DataFrame with imputed values
    """
    df = df.copy()

    # Impute numerical columns
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            if num_strategy == 'median':
                fill_value = df[col].median()
            elif num_strategy == 'mean':
                fill_value = df[col].mean()
            else:
                fill_value = 0
            df[col] = df[col].fillna(fill_value)

    # Impute categorical columns
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(cat_placeholder)

    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
    global_categories: Optional[Dict] = None,
    drop_first: bool = False
) -> pd.DataFrame:
    """
    One-hot encode categorical columns using global categories for consistency.

    Args:
        df: Input DataFrame
        categorical_cols: List of categorical column names to encode
        global_categories: Dict mapping column names to list of all possible categories
        drop_first: Whether to drop first category (for linear models)

    Returns:
        DataFrame with one-hot encoded columns
    """
    if global_categories is None:
        global_categories = GLOBAL_CATEGORIES

    df = df.copy()
    cols_to_encode = []

    for col in categorical_cols:
        if col not in df.columns:
            continue

        if col in global_categories:
            # Add -1.0 for missing value placeholder if not already present
            all_cats = list(global_categories[col])
            if -1.0 not in all_cats:
                all_cats = [-1.0] + all_cats

            # Convert to categorical with global categories
            df[col] = pd.Categorical(df[col], categories=all_cats)
            cols_to_encode.append(col)
        else:
            # Use observed categories if global not defined
            df[col] = df[col].astype('category')
            cols_to_encode.append(col)

    if cols_to_encode:
        df = pd.get_dummies(df, columns=cols_to_encode, drop_first=drop_first, dummy_na=False)

    return df


def scale_numerical(
    df: pd.DataFrame,
    numerical_cols: List[str],
    scaler: Optional[StandardScaler] = None
) -> tuple:
    """
    Scale numerical columns using StandardScaler.

    Args:
        df: Input DataFrame
        numerical_cols: List of numerical column names to scale
        scaler: Pre-fitted scaler (if None, creates and fits a new one)

    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    df = df.copy()
    cols_to_scale = [col for col in numerical_cols if col in df.columns]

    if not cols_to_scale:
        return df, scaler

    if scaler is None:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df, scaler


def preprocess_acs_data_with_scaler(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    drop_high_cardinality: bool = True,
    verbose: bool = False
) -> tuple:
    """
    Preprocess ACS data and return the scaler for later use (e.g., test data).

    Args:
        df: Input DataFrame
        scaler: Pre-fitted scaler (if None, creates and fits a new one)
        drop_high_cardinality: Whether to drop high-cardinality columns
        verbose: Whether to print progress information

    Returns:
        Tuple of (preprocessed DataFrame, fitted scaler)
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input must be a non-empty pandas DataFrame")

    df_processed = df.copy()

    # Identify feature types
    numerical_cols, categorical_cols, high_card_cols = get_feature_types(df_processed)

    # Drop high-cardinality columns
    if drop_high_cardinality and high_card_cols:
        df_processed = df_processed.drop(columns=high_card_cols, errors='ignore')

    # Impute missing values
    df_processed = impute_missing_values(df_processed, numerical_cols, categorical_cols)

    # One-hot encode categoricals
    if categorical_cols:
        df_processed = encode_categoricals(df_processed, categorical_cols)

    # Scale numerical features
    if numerical_cols:
        df_processed, scaler = scale_numerical(df_processed, numerical_cols, scaler)

    if verbose:
        print(f"Preprocessed shape: {df_processed.shape}")

    return df_processed, scaler


# =============================================================================
# Task features
# =============================================================================

# Per-task raw feature columns (before one-hot encoding).
TASK_FEATURES = {
    'ACSIncome': ['AGEP', 'COW', 'SCHL', 'MAR', 'RELP', 'WKHP', 'SEX', 'RAC1P'],
    'ACSEmployment': ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
                     'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P'],
    'ACSPublicCoverage': ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
                         'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'FER', 'RAC1P'],
    'ACSMobility': ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC',
                   'NATIVITY', 'RELP', 'DEAR', 'DEYE', 'DREM', 'RAC1P', 'GCL', 'COW', 'ESR', 'WKHP', 'JWMNP', 'PINCP'],
    'ACSTravelTime': ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'MIG', 'RELP', 'RAC1P',
                     'CIT', 'JWTR', 'POVPIP'],
}
