# -*- coding: utf-8 -*-
"""
Data Preprocessor Module for Credit Risk Alternative Data Analysis
==================================================================

This module handles all data preprocessing for the credit risk modeling pipeline.
It performs the following key operations:

1. Data Loading & Merging: Loads 7 CSV files and merges them by SK_ID_CURR
2. Feature Engineering: Creates domain-specific features (ratios, age, etc.)
3. Windowizing: Applies Yeo-Johnson power transformation to skewed features
4. Categorical Encoding: Binary LabelEncoding or Top-5 One-Hot encoding
5. Feature Separation: Splits features into Traditional vs Alternative categories
6. Train/Validation Split: 80/20 stratified split
7. SMOTE Balancing: Synthetic minority oversampling (50% ratio)
8. StandardScaler: Normalizes features to zero mean, unit variance

Data Flow:
----------
    Raw CSVs (2.6 GB) -> Merge -> Engineer -> Windowize -> Encode -> Split -> Balance -> Scale

Output:
-------
    - preprocessed_data.pkl: Contains 3 datasets (all, traditional, alternative)
    - preprocessor.pkl: Fitted preprocessor object for later use

Author: Credit Risk Analysis Project
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import warnings
import pickle
import gc
import os
from datetime import datetime
from typing import Optional, Dict, List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# =============================================================================
# DATA PREPROCESSOR CLASS
# =============================================================================
class DataPreprocessor:
    """
    Handles all data preprocessing including feature engineering, transformation,
    encoding, and balancing for credit risk modeling.

    This class is designed for the Home Credit Default Risk dataset, which contains
    multiple related tables that need to be merged and processed before modeling.

    Attributes
    ----------
    label_encoders : dict
        Stores fitted LabelEncoder objects for binary categorical columns.
        Keys are column names, values are fitted LabelEncoder instances.

    scaler : StandardScaler
        Primary scaler (note: each feature set gets its own scaler in practice).

    power_transformers : dict
        Stores fitted PowerTransformer objects for skewed numerical columns.
        Keys are column names, values are fitted PowerTransformer instances.
        These are saved for applying the same transformation to test data.

    alternative_features : list
        List of column names classified as "alternative" data features.
        These include: FLAG_*, EXT_SOURCE*, REGION_*, contact flags, etc.

    traditional_features : list
        List of column names classified as "traditional" credit features.
        These include: AMT_*, DAYS_*, bureau history, payment history, etc.

    all_features : list
        Complete list of all feature column names after preprocessing.

    Example Usage
    -------------
    >>> from src.pipeline.data_preprocessor import DataPreprocessor
    >>> from src.utils.paths import data_path
    >>>
    >>> train_paths = {
    ...     'application': data_path('application_train.csv'),
    ...     'bureau': data_path('bureau.csv'),
    ...     # ... other paths
    ... }
    >>>
    >>> preprocessor = DataPreprocessor()
    >>> datasets = preprocessor.preprocess_and_save(train_paths)
    >>>
    >>> # Access preprocessed data
    >>> X_train = datasets['all']['X_train']
    >>> y_train = datasets['all']['y_train']
    """

    def __init__(self):
        """
        Initialize the DataPreprocessor with empty containers for fitted transformers.

        All transformers are fitted during preprocess_and_save() and stored for
        potential reuse on new data.
        """
        # Fitted transformers storage
        self.label_encoders = {}      # For binary categorical encoding
        self.scaler = StandardScaler()  # Default scaler instance
        self.power_transformers = {}  # For Yeo-Johnson transformations

        # Feature categorization storage
        self.alternative_features = []  # Non-traditional data features
        self.traditional_features = []  # Standard credit bureau features
        self.all_features = []          # All features after preprocessing


    def load_and_merge_data(self, data_paths: Dict[str, str]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and merge all datasets from the Home Credit competition.

        This method loads 7 CSV files and merges them into a single DataFrame
        at the customer level (SK_ID_CURR). Secondary tables are aggregated
        using mean, max, and min statistics before merging.

        Parameters
        ----------
        data_paths : dict
            Dictionary mapping data types to file paths.
            Required key: 'application' (main application data)
            Optional keys: 'bureau', 'bureau_balance', 'previous_application',
                          'credit_card_balance', 'pos_cash_balance', 'installments_payments'

        Returns
        -------
        tuple of (DataFrame, ndarray)
            - Merged DataFrame with all features
            - Target array (y) or None if no TARGET column exists

        Data Relationships
        ------------------
        application_train (SK_ID_CURR)
            |
            +-> bureau (SK_ID_CURR -> SK_ID_BUREAU)
            |       |
            |       +-> bureau_balance (SK_ID_BUREAU)
            |
            +-> previous_application (SK_ID_CURR -> SK_ID_PREV)
                    |
                    +-> credit_card_balance (SK_ID_PREV)
                    +-> POS_CASH_balance (SK_ID_PREV)
                    +-> installments_payments (SK_ID_PREV)
        """
        print("Loading datasets...")

        # =====================================================================
        # Step 1: Load main application data (required)
        # This is the primary table containing loan applications with TARGET
        # =====================================================================
        app_data = pd.read_csv(data_paths['application'])
        print(f"  Application data shape: {app_data.shape}")

        # Extract target variable if present (training data has TARGET, test doesn't)
        if 'TARGET' in app_data.columns:
            target = app_data['TARGET'].values
        else:
            target = None

        # Apply feature engineering to application data first
        app_data = self.engineer_basic_features(app_data)

        # =====================================================================
        # Step 2: Process Bureau Data
        # Contains credit history from other financial institutions
        # =====================================================================
        if 'bureau' in data_paths and os.path.exists(data_paths['bureau']):
            print("  Loading bureau data...")
            bureau = pd.read_csv(data_paths['bureau'])

            # Merge bureau_balance (monthly snapshots) if available
            if 'bureau_balance' in data_paths and os.path.exists(data_paths['bureau_balance']):
                print("  Loading bureau_balance data...")
                bureau_balance = pd.read_csv(data_paths['bureau_balance'])

                # Aggregate bureau_balance to bureau level (SK_ID_BUREAU)
                # This creates features like: how long the credit was tracked,
                # and how many months had "good" status (STATUS='0')
                bb_agg = bureau_balance.groupby('SK_ID_BUREAU').agg({
                    'MONTHS_BALANCE': ['min', 'max', 'mean'],  # Credit tracking duration
                    'STATUS': lambda x: (x == '0').sum()        # Count of "good" months
                }).fillna(0)
                bb_agg.columns = ['BB_' + '_'.join(col).strip() for col in bb_agg.columns.values]
                bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')

            # Aggregate bureau to customer level (SK_ID_CURR)
            # Use mean, max, min for each numeric feature
            bureau_numeric = bureau.select_dtypes(include=[np.number])
            bureau_agg = bureau_numeric.groupby('SK_ID_CURR').agg({
                col: ['mean', 'max', 'min'] for col in bureau_numeric.columns
                if col not in ['SK_ID_CURR', 'SK_ID_BUREAU']
            }).fillna(0)
            bureau_agg.columns = ['BUREAU_' + '_'.join(col).strip() for col in bureau_agg.columns.values]
            app_data = app_data.merge(bureau_agg, on='SK_ID_CURR', how='left')

        # =====================================================================
        # Step 3: Process Previous Application Data
        # Contains history of prior loan applications at Home Credit
        # =====================================================================
        if 'previous_application' in data_paths and os.path.exists(data_paths['previous_application']):
            print("  Loading previous application data...")
            prev_app = pd.read_csv(data_paths['previous_application'])
            prev_numeric = prev_app.select_dtypes(include=[np.number])
            prev_agg = prev_numeric.groupby('SK_ID_CURR').agg({
                col: ['mean', 'max', 'min'] for col in prev_numeric.columns
                if col not in ['SK_ID_CURR', 'SK_ID_PREV']
            }).fillna(0)
            prev_agg.columns = ['PREV_' + '_'.join(col).strip() for col in prev_agg.columns.values]
            app_data = app_data.merge(prev_agg, on='SK_ID_CURR', how='left')

        # =====================================================================
        # Step 4: Process Credit Card Balance Data
        # Contains monthly balance snapshots for previous credit cards
        # =====================================================================
        if 'credit_card_balance' in data_paths and os.path.exists(data_paths['credit_card_balance']):
            print("  Loading credit card balance data...")
            cc_balance = pd.read_csv(data_paths['credit_card_balance'])
            cc_numeric = cc_balance.select_dtypes(include=[np.number])
            cc_agg = cc_numeric.groupby('SK_ID_CURR').agg({
                col: ['mean', 'max', 'min'] for col in cc_numeric.columns
                if col not in ['SK_ID_CURR', 'SK_ID_PREV']
            }).fillna(0)
            cc_agg.columns = ['CC_' + '_'.join(col).strip() for col in cc_agg.columns.values]
            app_data = app_data.merge(cc_agg, on='SK_ID_CURR', how='left')

        # =====================================================================
        # Step 5: Process POS Cash Balance Data
        # Contains monthly snapshots for POS (point of sale) and cash loans
        # =====================================================================
        if 'pos_cash_balance' in data_paths and os.path.exists(data_paths['pos_cash_balance']):
            print("  Loading POS cash balance data...")
            pos_balance = pd.read_csv(data_paths['pos_cash_balance'])
            pos_numeric = pos_balance.select_dtypes(include=[np.number])
            pos_agg = pos_numeric.groupby('SK_ID_CURR').agg({
                col: ['mean', 'max', 'min'] for col in pos_numeric.columns
                if col not in ['SK_ID_CURR', 'SK_ID_PREV']
            }).fillna(0)
            pos_agg.columns = ['POS_' + '_'.join(col).strip() for col in pos_agg.columns.values]
            app_data = app_data.merge(pos_agg, on='SK_ID_CURR', how='left')

        # =====================================================================
        # Step 6: Process Installments Payments Data
        # Contains payment history for previous loans - very valuable for
        # understanding payment behavior patterns
        # =====================================================================
        if 'installments_payments' in data_paths and os.path.exists(data_paths['installments_payments']):
            print("  Loading installments payments data...")
            installments = pd.read_csv(data_paths['installments_payments'])

            # Engineer payment behavior features
            # PAYMENT_DIFF: Positive = overpaid, Negative = underpaid
            # PAYMENT_RATIO: >1 = overpaid, <1 = underpaid
            installments['PAYMENT_DIFF'] = installments['AMT_PAYMENT'] - installments['AMT_INSTALMENT']
            installments['PAYMENT_RATIO'] = installments['AMT_PAYMENT'] / (installments['AMT_INSTALMENT'] + 0.0001)

            inst_numeric = installments.select_dtypes(include=[np.number])
            inst_agg = inst_numeric.groupby('SK_ID_CURR').agg({
                col: ['mean', 'max', 'min'] for col in inst_numeric.columns
                if col not in ['SK_ID_CURR', 'SK_ID_PREV']
            }).fillna(0)
            inst_agg.columns = ['INST_' + '_'.join(col).strip() for col in inst_agg.columns.values]
            app_data = app_data.merge(inst_agg, on='SK_ID_CURR', how='left')

        print(f"  Final merged shape: {app_data.shape}")
        return app_data, target


    def engineer_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific engineered features from raw application data.

        These features capture important credit risk concepts like debt burden,
        income stability, and external credit assessments.

        Parameters
        ----------
        df : DataFrame
            Raw application data with original columns

        Returns
        -------
        DataFrame
            Application data with additional engineered features

        Engineered Features
        -------------------
        Credit Ratios (measure debt burden):
            - CREDIT_INCOME_RATIO: Loan amount relative to income
            - ANNUITY_INCOME_RATIO: Monthly payment relative to income
            - CREDIT_GOODS_RATIO: Loan-to-value ratio

        Time Features (derived from DAYS_* columns):
            - AGE_YEARS: Customer age in years
            - EMPLOYMENT_YEARS: Employment duration in years
            - DAYS_EMPLOYED_PERCENT: Employment duration as % of age

        External Source Aggregates (combine multiple external scores):
            - EXT_SOURCE_MEAN: Average of external scores
            - EXT_SOURCE_STD: Variability in external scores
            - EXT_SOURCE_MIN: Worst external score
            - EXT_SOURCE_MAX: Best external score
        """
        # -----------------------------------------------------------------
        # Credit Ratios: Measure debt burden relative to income/assets
        # -----------------------------------------------------------------
        if 'AMT_CREDIT' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            # How many times the annual income is the loan amount?
            # Higher values indicate higher leverage/risk
            df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)

        if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
            # What fraction of income goes to loan payments?
            # Similar to debt-to-income ratio
            df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)

        if 'AMT_CREDIT' in df.columns and 'AMT_GOODS_PRICE' in df.columns:
            # Loan-to-value ratio: Is the loan larger than the goods purchased?
            # Values > 1 might indicate additional fees or insurance included
            df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)

        # -----------------------------------------------------------------
        # Time Features: Convert DAYS_* (negative days) to human-readable
        # -----------------------------------------------------------------
        if 'DAYS_BIRTH' in df.columns:
            # DAYS_BIRTH is negative (days before application)
            # Convert to positive years
            df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25

        if 'DAYS_EMPLOYED' in df.columns:
            # DAYS_EMPLOYED is negative for employed, 365243 for unemployed/retired
            # Convert to years and clip negative values (unemployed) to 0
            df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25
            df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].clip(lower=0)

        if 'DAYS_EMPLOYED' in df.columns and 'DAYS_BIRTH' in df.columns:
            # Employment as percentage of life - stability indicator
            df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)

        # -----------------------------------------------------------------
        # External Source Aggregates: Combine multiple external credit scores
        # EXT_SOURCE features are the most predictive in this dataset
        # -----------------------------------------------------------------
        ext_source_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        ext_source_cols = [col for col in ext_source_cols if col in df.columns]
        if len(ext_source_cols) > 0:
            df['EXT_SOURCE_MEAN'] = df[ext_source_cols].mean(axis=1)  # Average score
            df['EXT_SOURCE_STD'] = df[ext_source_cols].std(axis=1)    # Score variability
            df['EXT_SOURCE_MIN'] = df[ext_source_cols].min(axis=1)    # Worst score
            df['EXT_SOURCE_MAX'] = df[ext_source_cols].max(axis=1)    # Best score

        return df


    def apply_windowizing(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Apply Yeo-Johnson power transformation to skewed numerical features.

        "Windowizing" refers to the process of making distributions more
        Gaussian-like, which improves the performance of many ML algorithms,
        especially linear models.

        Parameters
        ----------
        df : DataFrame
            Data with numerical features that may be skewed
        threshold : float, default=0.5
            Skewness threshold. Features with |skewness| > threshold
            will be transformed.

        Returns
        -------
        DataFrame
            Data with transformed features

        Why Yeo-Johnson?
        ----------------
        - Works with both positive and negative values (unlike Box-Cox)
        - Automatically selects optimal transformation parameter (lambda)
        - Makes distributions more Gaussian-like
        - Reduces impact of outliers
        - Improves linear model performance

        Notes
        -----
        The fitted PowerTransformer for each column is stored in
        self.power_transformers for applying to test data later.
        """
        print("  Applying windowizing (Yeo-Johnson transformation) to skewed features...")

        # Target columns likely to be skewed (financial amounts, ratios)
        target_cols = ['AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE',
                      'AMT_ANNUITY', 'DAYS_EMPLOYED', 'CREDIT_INCOME_RATIO',
                      'ANNUITY_INCOME_RATIO', 'CREDIT_GOODS_RATIO']

        # Also include all AMT_ columns (amounts are typically right-skewed)
        amt_cols = [col for col in df.columns if 'AMT_' in col]
        target_cols.extend(amt_cols)
        target_cols = list(set(target_cols))  # Remove duplicates

        transformed_cols = []

        for col in target_cols:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                # Calculate skewness: 0 = symmetric, positive = right-skewed
                skewness = df[col].skew()

                # Only transform if highly skewed
                if abs(skewness) > threshold:
                    try:
                        # Yeo-Johnson can handle zero and negative values
                        pt = PowerTransformer(method='yeo-johnson', standardize=False)
                        df[col] = pt.fit_transform(df[[col]].values.reshape(-1, 1)).flatten()
                        self.power_transformers[col] = pt  # Save for test data
                        transformed_cols.append(col)
                    except:
                        # Skip if transformation fails (e.g., constant column)
                        continue

        print(f"    Transformed {len(transformed_cols)} skewed features")
        if len(transformed_cols) > 0 and len(transformed_cols) <= 15:
            print(f"    Features: {', '.join(transformed_cols[:15])}")

        return df


    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using appropriate strategies.

        Strategy:
        - Binary columns (2 unique values): LabelEncoder (0/1)
        - Multi-category columns: Top-5 One-Hot Encoding

        Parameters
        ----------
        df : DataFrame
            Data with categorical (object type) columns

        Returns
        -------
        DataFrame
            Data with encoded categorical features

        Why Top-5 One-Hot?
        ------------------
        - Reduces dimensionality compared to full one-hot encoding
        - Captures the most important categories
        - Rare categories often don't have predictive power
        - Prevents curse of dimensionality

        Notes
        -----
        Fitted LabelEncoders are stored in self.label_encoders for
        applying to test data later.
        """
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"  Encoding {len(categorical_cols)} categorical columns...")

        for col in categorical_cols:
            # Skip ID columns
            if col in ['SK_ID_CURR', 'SK_ID_PREV']:
                continue

            if df[col].nunique() <= 2:
                # Binary encoding using LabelEncoder
                # Examples: FLAG_OWN_CAR (Y/N), CODE_GENDER (M/F)
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna('Missing'))
                self.label_encoders[col] = le
            else:
                # Top-5 One-Hot encoding for multi-category columns
                # This limits dimensionality while capturing key categories
                top_cats = df[col].value_counts().head(5).index.tolist()
                for cat in top_cats:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                df = df.drop(col, axis=1)

        return df


    def separate_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Separate features into Alternative and Traditional categories.

        This separation enables comparing model performance with different
        types of data sources - a key goal of this research project.

        Parameters
        ----------
        df : DataFrame
            Data with all features after preprocessing

        Returns
        -------
        tuple of (list, list)
            - alternative_features: List of alternative data feature names
            - traditional_features: List of traditional credit feature names

        Feature Categories
        ------------------
        Alternative Features (non-traditional data):
            - FLAG_*: Document and contact flags
            - EXT_SOURCE*: External credit scores
            - REGION_*: Geographic indicators
            - OBS_*, DEF_*: Observation/default counters
            - Contact flags: EMAIL, PHONE, MOBIL

        Traditional Features (standard credit bureau data):
            - AMT_*: Loan and income amounts
            - DAYS_*: Time-based features
            - CNT_*: Count features
            - BUREAU_*: External credit bureau history
            - PREV_*: Previous application history
            - Credit/Income related features
        """
        # Keywords indicating alternative (non-traditional) data sources
        alternative_keywords = ['FLAG_', 'EXT_SOURCE', 'REGION_', 'OBS_', 'DEF_',
                              'EMAIL', 'PHONE', 'MOBIL', 'SOCIAL']

        # Keywords indicating traditional credit bureau data
        traditional_keywords = ['AMT_', 'DAYS_', 'CNT_', 'CREDIT', 'INCOME',
                               'BUREAU_', 'PREV_', 'ANNUITY']

        for col in df.columns:
            # Skip non-feature columns
            if col in ['TARGET', 'SK_ID_CURR']:
                continue

            # Check if feature matches alternative or traditional keywords
            is_alternative = any(keyword in col.upper() for keyword in alternative_keywords)
            is_traditional = any(keyword in col.upper() for keyword in traditional_keywords)

            # Classify feature (traditional takes precedence if both match)
            if is_alternative and not is_traditional:
                self.alternative_features.append(col)
            else:
                self.traditional_features.append(col)

        self.all_features = df.columns.tolist()

        return self.alternative_features, self.traditional_features


    def preprocess_and_save(self, train_paths: Dict[str, str],
                           test_paths: Optional[Dict[str, str]] = None) -> Dict:
        """
        Main preprocessing function that executes the complete pipeline.

        This method orchestrates all preprocessing steps and saves the
        results for model training.

        Parameters
        ----------
        train_paths : dict
            Dictionary mapping data types to training file paths
        test_paths : dict, optional
            Dictionary mapping data types to test file paths (not used currently)

        Returns
        -------
        dict
            Dictionary with three keys: 'all', 'traditional', 'alternative'
            Each contains:
            - X_train: Scaled, balanced training features (numpy array)
            - X_val: Scaled validation features (numpy array)
            - y_train: Balanced training labels (numpy array)
            - y_val: Validation labels (numpy array)
            - features: List of feature names
            - scaler: Fitted StandardScaler

        Pipeline Steps
        --------------
        1. Load & merge all CSV files
        2. Clean data (handle inf/nan)
        3. Apply Yeo-Johnson transformation (windowizing)
        4. Encode categorical variables
        5. Separate alternative vs traditional features
        6. Train/validation split (80/20, stratified)
        7. Apply SMOTE balancing (50% ratio)
        8. Scale features with StandardScaler
        9. Save preprocessed data to pickle files

        Output Files
        ------------
        - preprocessed_data.pkl: Contains all preprocessed datasets
        - preprocessor.pkl: Fitted preprocessor object
        """
        print("\n" + "="*60)
        print("STEP 1: DATA PREPROCESSING")
        print("="*60)

        # Step 1: Load and merge all data sources
        print("\n1.1 Processing training data...")
        train_data, train_target = self.load_and_merge_data(train_paths)

        # Step 2: Clean data - handle infinite and missing values
        print("1.2 Cleaning data...")
        train_data = train_data.replace([np.inf, -np.inf], np.nan)
        train_data = train_data.fillna(0)  # Fill missing with 0

        # Step 3: Apply windowizing (power transformation)
        # IMPORTANT: Do this BEFORE encoding to transform numerical features
        print("1.3 WINDOWIZING - Power transformation for skewed features...")
        train_data = self.apply_windowizing(train_data)

        # Step 4: Encode categorical features
        print("1.4 Encoding categorical features...")
        train_data = self.encode_categorical(train_data)

        # Step 5: Separate features into traditional vs alternative
        print("1.5 Separating feature types...")
        alt_features, trad_features = self.separate_features(train_data)
        print(f"  Alternative features: {len(alt_features)}")
        print(f"  Traditional features: {len(trad_features)}")

        # Prepare feature matrix and target
        X = train_data.drop(['SK_ID_CURR', 'TARGET'], axis=1, errors='ignore')
        y = train_target

        # Step 6: Train/validation split (stratified to maintain class balance)
        print("1.6 Splitting train/validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Step 7: Setup SMOTE for class balancing
        # 50% ratio means minority class becomes 50% of majority class
        # Original: ~8% default -> After SMOTE: ~33% default
        print("1.7 Applying SMOTE (50% ratio to save memory)...")
        smote = SMOTE(random_state=42, sampling_strategy=0.5)

        # Create datasets dictionary to store all feature sets
        datasets = {}

        # -----------------------------------------------------------------
        # Process ALL FEATURES (381 features)
        # -----------------------------------------------------------------
        scaler_all = StandardScaler()
        X_train_scaled = scaler_all.fit_transform(X_train)
        X_val_scaled = scaler_all.transform(X_val)  # Use same scaler!
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train.astype(int))

        datasets['all'] = {
            'X_train': X_train_balanced,
            'X_val': X_val_scaled,
            'y_train': y_train_balanced,
            'y_val': y_val,
            'features': X.columns.tolist(),
            'scaler': scaler_all
        }

        # -----------------------------------------------------------------
        # Process TRADITIONAL FEATURES ONLY (334 features)
        # Standard credit bureau data that has been used for decades
        # -----------------------------------------------------------------
        if len(trad_features) > 0:
            trad_cols = [col for col in X.columns if col in trad_features]
            if len(trad_cols) > 0:
                X_train_trad = X_train[trad_cols]
                X_val_trad = X_val[trad_cols]

                scaler_trad = StandardScaler()
                X_train_trad_scaled = scaler_trad.fit_transform(X_train_trad)
                X_val_trad_scaled = scaler_trad.transform(X_val_trad)
                X_train_trad_balanced, y_train_trad_balanced = smote.fit_resample(
                    X_train_trad_scaled, y_train.astype(int)
                )

                datasets['traditional'] = {
                    'X_train': X_train_trad_balanced,
                    'X_val': X_val_trad_scaled,
                    'y_train': y_train_trad_balanced,
                    'y_val': y_val,
                    'features': trad_cols,
                    'scaler': scaler_trad
                }

        # -----------------------------------------------------------------
        # Process ALTERNATIVE FEATURES ONLY (47 features)
        # Non-traditional data that can help thin-file customers
        # -----------------------------------------------------------------
        if len(alt_features) > 0:
            alt_cols = [col for col in X.columns if col in alt_features]
            if len(alt_cols) > 0:
                X_train_alt = X_train[alt_cols]
                X_val_alt = X_val[alt_cols]

                scaler_alt = StandardScaler()
                X_train_alt_scaled = scaler_alt.fit_transform(X_train_alt)
                X_val_alt_scaled = scaler_alt.transform(X_val_alt)
                X_train_alt_balanced, y_train_alt_balanced = smote.fit_resample(
                    X_train_alt_scaled, y_train.astype(int)
                )

                datasets['alternative'] = {
                    'X_train': X_train_alt_balanced,
                    'X_val': X_val_alt_scaled,
                    'y_train': y_train_alt_balanced,
                    'y_val': y_val,
                    'features': alt_cols,
                    'scaler': scaler_alt
                }

        # Step 9: Save preprocessed data
        print("\n1.8 Saving preprocessed data...")
        with open('preprocessed_data.pkl', 'wb') as f:
            pickle.dump(datasets, f)

        # Save the preprocessor itself (for applying same transforms to new data)
        with open('preprocessor.pkl', 'wb') as f:
            pickle.dump(self, f)

        print("[OK] Preprocessing complete! Saved to 'preprocessed_data.pkl'")
        print(f"\n   Dataset sizes:")
        for name, data in datasets.items():
            print(f"   - {name}: Train {data['X_train'].shape}, Val {data['X_val'].shape}")

        # Clear memory
        del train_data, X, y, X_train, X_val
        gc.collect()

        return datasets
