"""
Data Preprocessing Module for Ensemble Deep Learning (EDL) Diabetes Prediction System

This module implements data preprocessing as described in the paper:
"An Innovative Ensemble Deep Learning Clinical Decision Support System for Diabetes Prediction"

Key functionalities:
1. Load three diabetes datasets (PIMA-IDD-I, DDFH-G, IDPD-I)
2. Clean datasets (remove missing/null/duplicate values)
3. Handle unrealistic zero values by replacing with feature mean
4. Normalize features using MinMaxScaler
5. Apply SMOTE for class imbalance handling
6. Split data into 80:20 train:test ratio
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import warnings
warnings.filterwarnings('ignore')

class DiabetesDataPreprocessor:
    def add_polynomial_features(self, df, dataset_name, degree=2):
        """
        Add polynomial and interaction features for PIMA dataset
        Args:
            df (pd.DataFrame): Input dataset
            dataset_name (str): Dataset identifier
            degree (int): Degree of polynomial features
        Returns:
            pd.DataFrame: Dataset with added features
        """
        if dataset_name != 'PIMA':
            return df
        print(f"\nAdding polynomial and interaction features to {dataset_name} dataset...")
        from sklearn.preprocessing import PolynomialFeatures
        feature_cols = df.columns[:-1]  # All except target
        poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
        poly_features = poly.fit_transform(df[feature_cols])
        poly_feature_names = poly.get_feature_names_out(feature_cols)
        df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        # Keep target column
        df_poly[df.columns[-1]] = df[df.columns[-1]]
        print(f"Added {df_poly.shape[1] - len(feature_cols) - 1} new features (degree={degree})")
        return df_poly
    """
    Data preprocessing class implementing the paper's methodology
    """
    
    def __init__(self):
        self.scalers = {}
        self.feature_columns = {}
        self.target_columns = {}
        self.datasets_info = {
            'PIMA': {
                'file': 'data/diabetes.csv',  # PIMA Indian Diabetes Dataset
                'instances': 768,
                'features': 9,
                'type': 'binary'
            },
            'DDFH': {
                'file': 'data/Dataset of Diabetes .csv',  # Frankfurt Hospital Germany
                'instances': 2000,
                'features': 8,
                'type': 'binary'
            },
            'IDPD': {
                'file': 'data/pima.csv.csv',  # Iraqi Dataset (using available data)
                'instances': 1000,
                'features': 11,
                'type': 'multi-class'
            },
            'COMBINED': {
                'file': 'data/combined_diabetes.csv',
                'instances': 3770,
                'features': 21,
                'type': 'binary'
            }
        }
    
    def load_dataset(self, dataset_name):
        """
        Load specified dataset
        
        Args:
            dataset_name (str): 'PIMA', 'DDFH', or 'IDPD'
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        if dataset_name not in self.datasets_info:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {list(self.datasets_info.keys())}")
        
        file_path = self.datasets_info[dataset_name]['file']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        print(f"Loading {dataset_name} dataset from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {dataset_name}: {df.shape[0]} instances, {df.shape[1]} features")
            return df
        except Exception as e:
            print(f"Error loading {dataset_name}: {str(e)}")
            return None
    
    def clean_dataset(self, df, dataset_name):
        """
        Clean dataset by removing missing/null/duplicate values
        
        Args:
            df (pd.DataFrame): Input dataset
            dataset_name (str): Dataset identifier
        
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        print(f"\nCleaning {dataset_name} dataset...")
        initial_shape = df.shape
        print(f"Initial shape: {initial_shape}")

        # Special handling for COMBINED dataset - it has mixed structure
        if dataset_name == 'COMBINED':
            # Analyze null patterns first
            null_percentages = df.isnull().sum() / len(df) * 100
            good_columns = null_percentages[null_percentages < 50].index.tolist()
            
            if good_columns:
                print(f"COMBINED: Using {len(good_columns)} columns with <50% missing data")
                print(f"Selected columns: {good_columns}")
                df = df[good_columns]
                
                # Remove rows with too many missing values (>50% of remaining columns)
                missing_threshold = len(good_columns) * 0.5
                df_clean = df.dropna(thresh=len(good_columns) - missing_threshold)
                print(f"After removing rows with >50% missing values: {df_clean.shape}")
                
                # Fill remaining missing values with median for numerical, mode for categorical
                for col in df_clean.columns:
                    if df_clean[col].isnull().sum() > 0:
                        if df_clean[col].dtype in ['object']:
                            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                        else:
                            df_clean[col].fillna(df_clean[col].median(), inplace=True)
                print(f"After filling remaining missing values: {df_clean.shape}")
            else:
                print("⚠️ No usable columns found in COMBINED dataset")
                return df.iloc[:0]  # Return empty dataframe
        else:
            # Original logic for other datasets
            df_clean = df.dropna()
            print(f"After removing null values: {df_clean.shape}")

        # Remove duplicate rows
        df_clean = df_clean.drop_duplicates()
        print(f"After removing duplicates: {df_clean.shape}")

        # Remove entries with all zeros (if any)
        df_clean = df_clean.loc[~(df_clean == 0).all(axis=1)]
        print(f"After removing all-zero rows: {df_clean.shape}")

        # For COMBINED dataset, drop columns that are completely empty and ensure only one valid target column is used
        if dataset_name == 'COMBINED':
            # Drop columns that are all NaN or all empty
            before_cols = df_clean.shape[1]
            df_clean = df_clean.dropna(axis=1, how='all')
            df_clean = df_clean.loc[:, ~(df_clean == '').all(axis=0)]
            after_cols = df_clean.shape[1]
            print(f"Dropped {before_cols - after_cols} completely empty columns.")

            # Try both possible target columns, but keep only one as target
            target_cols = ['Outcome', 'CLASS']
            found_target = None
            for col in target_cols:
                if col in df_clean.columns:
                    before = df_clean.shape[0]
                    df_clean = df_clean[df_clean[col].notnull() & (df_clean[col] != '')]
                    after = df_clean.shape[0]
                    print(f"After dropping rows with missing/empty '{col}': {after} rows (removed {before - after})")
                    found_target = col
                    break
            # Drop the other target column if both exist
            for col in target_cols:
                if col != found_target and col in df_clean.columns:
                    df_clean = df_clean.drop(columns=[col])

        print(f"Cleaned {dataset_name}: Removed {initial_shape[0] - df_clean.shape[0]} rows")
        return df_clean
    
    def handle_zero_values(self, df, dataset_name):
        """
        Handle unrealistic zero values by replacing with feature mean
        Also handles categorical encoding for DDFH dataset
        
        Args:
            df (pd.DataFrame): Input dataset
            dataset_name (str): Dataset identifier
        
        Returns:
            pd.DataFrame: Dataset with zero values handled and categorical variables encoded
        """
        print(f"\nHandling zero values and categorical encoding in {dataset_name} dataset...")
        
        df_processed = df.copy()
        
        # Handle categorical variables first (specific to DDFH dataset)
        if 'Gender' in df_processed.columns:
            print("  Encoding categorical 'Gender' column...")
            # Check current values
            print(f"    Current Gender values: {df_processed['Gender'].unique()}")
            # Encode Gender: M -> 1, F -> 0
            gender_mapping = {'M': 1, 'F': 0}
            # Handle any case variations
            df_processed['Gender'] = df_processed['Gender'].str.upper().map(gender_mapping)
            print(f"    Gender encoding complete. New values: {df_processed['Gender'].unique()}")
        
        if 'CLASS' in df_processed.columns:
            print("  Encoding categorical 'CLASS' column...")
            # Check current values
            print(f"    Current CLASS values: {df_processed['CLASS'].unique()}")
            # Encode CLASS: N -> 0 (No diabetes), P -> 1 (Pre-diabetes), Y -> 2 (Yes/diabetes)
            class_mapping = {'N': 0, 'P': 1, 'Y': 2}
            # Handle any case variations and strip whitespace
            df_processed['CLASS'] = df_processed['CLASS'].astype(str).str.strip().str.upper().map(class_mapping)
            print(f"    CLASS encoding complete. New values: {df_processed['CLASS'].unique()}")
            print(f"    CLASS distribution: {df_processed['CLASS'].value_counts().to_dict()}")
        
        # Define features that should not have zero values based on dataset
        zero_handle_features = []
        
        if dataset_name == 'PIMA':
            # PIMA dataset - common features that shouldn't be zero
            pima_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            zero_handle_features = [f for f in pima_features if f in df_processed.columns]
            print(f"  PIMA dataset: handling zero values in {zero_handle_features}")
            
        elif dataset_name == 'DDFH':
            # DDFH dataset - be more selective, as some medical values can legitimately be zero
            # Only handle BMI as it should not be zero; other medical values might legitimately be zero
            ddfh_features = ['BMI']
            zero_handle_features = [f for f in ddfh_features if f in df_processed.columns]
            print(f"  DDFH dataset: handling zero values in {zero_handle_features}")
            
        else:
            # For other datasets, check all numeric columns except target
            for col in df_processed.columns[:-1]:  # Don't process target column
                if df_processed[col].dtype in ['int64', 'float64']:  # Only numeric columns
                    if (df_processed[col] == 0).sum() > 0:  # Has zero values
                        zero_handle_features.append(col)
            print(f"  {dataset_name} dataset: handling zero values in {zero_handle_features}")
        
        zero_handle_features = list(set(zero_handle_features))  # Remove duplicates
        
        print(f"Features to handle zero values: {zero_handle_features}")
        
        for feature in zero_handle_features:
            if feature in df_processed.columns:
                zero_count = (df_processed[feature] == 0).sum()
                if zero_count > 0:
                    mean_value = df_processed[df_processed[feature] != 0][feature].mean()
                    df_processed.loc[df_processed[feature] == 0, feature] = mean_value
                    print(f"Replaced {zero_count} zero values in '{feature}' with mean: {mean_value:.2f}")
        
        return df_processed
    
    def normalize_features(self, df, dataset_name, fit_scaler=True):
        """
        Normalize features using MinMaxScaler
        
        Args:
            df (pd.DataFrame): Input dataset
            dataset_name (str): Dataset identifier
            fit_scaler (bool): Whether to fit new scaler or use existing
        
        Returns:
            pd.DataFrame: Normalized dataset
        """
        print(f"\nNormalizing features for {dataset_name} dataset...")
        
        df_normalized = df.copy()

        # Check for empty DataFrame
        if df.shape[0] == 0 or df.shape[1] == 0:
            print(f"⚠️ Warning: DataFrame is empty after cleaning for dataset {dataset_name}.")
            print(f"This might be due to:")
            print(f"  1. Dataset has too many missing values")
            print(f"  2. Aggressive cleaning parameters")
            print(f"  3. Mixed dataset structure (try individual datasets)")
            print(f"💡 Recommendation: Use PIMA or DDFH datasets individually for best results.")
            raise ValueError(f"Cannot proceed with empty dataset {dataset_name}. Consider using individual datasets.")

        # Separate features and target
        feature_cols = df.columns[:-1]  # All columns except last (target)
        target_col = df.columns[-1]     # Last column (target)

        self.feature_columns[dataset_name] = list(feature_cols)
        self.target_columns[dataset_name] = target_col

        # Normalize features
        if fit_scaler:
            scaler = MinMaxScaler()
            df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scalers[dataset_name] = scaler
            print(f"Fitted and applied MinMaxScaler for {dataset_name}")
        else:
            if dataset_name in self.scalers:
                df_normalized[feature_cols] = self.scalers[dataset_name].transform(df[feature_cols])
                print(f"Applied existing MinMaxScaler for {dataset_name}")
            else:
                raise ValueError(f"No fitted scaler found for {dataset_name}")

        return df_normalized
    
    def apply_smote(self, X, y, dataset_name):
        """
        Apply SMOTE for class imbalance handling
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            dataset_name (str): Dataset identifier
        
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        print(f"\nApplying SMOTE to {dataset_name} dataset...")
        
        # Print original class distribution
        print("Original class distribution:")
        print(y.value_counts().sort_index())
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Print new class distribution
        print("After SMOTE class distribution:")
        unique, counts = np.unique(y_resampled, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"Class {val}: {count}")
        
        print(f"Dataset size increased from {len(X)} to {len(X_resampled)} instances")
        
        return X_resampled, y_resampled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets (80:20 ratio)
        
        Args:
            X: Features
            y: Target
            test_size (float): Test set proportion (default: 0.2 for 80:20 split)
            random_state (int): Random state for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"\nSplitting data into {int((1-test_size)*100)}:{int(test_size*100)} train:test ratio...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} instances")
        print(f"Test set: {X_test.shape[0]} instances")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_dataset(self, dataset_name, apply_smote_flag=True):
        """
        Complete preprocessing pipeline for a dataset
        
        Args:
            dataset_name (str): 'PIMA', 'DDFH', or 'IDPD'
            apply_smote_flag (bool): Whether to apply SMOTE
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, dataset_info)
        """
        print(f"\n{'='*60}")
        print(f"PREPROCESSING {dataset_name} DATASET")
        print(f"{'='*60}")
        
        # Step 1: Load dataset
        df = self.load_dataset(dataset_name)
        if df is None:
            return None
        
        # Step 2: Clean dataset
        df_clean = self.clean_dataset(df, dataset_name)
        
        # Step 3: Handle zero values
        df_processed = self.handle_zero_values(df_clean, dataset_name)
        
        # Step 3.5: Add polynomial and interaction features for PIMA
        df_poly = self.add_polynomial_features(df_processed, dataset_name, degree=2)
        
        # Step 4: Normalize features
        df_normalized = self.normalize_features(df_poly, dataset_name, fit_scaler=True)
        
        # Step 5: Separate features and target
        feature_cols = self.feature_columns[dataset_name]
        target_col = self.target_columns[dataset_name]
        
        X = df_normalized[feature_cols]
        y = df_normalized[target_col]
        
        # Step 6: Apply SMOTE if requested
        if apply_smote_flag:
            X, y = self.apply_smote(X, y, dataset_name)
        
        # Step 7: Split data (80:20)
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Dataset info for reference
        dataset_info = {
            'name': dataset_name,
            'original_shape': df.shape,
            'processed_shape': (len(X), len(feature_cols)),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': feature_cols,
            'target': target_col,
            'classes': sorted(y.unique())
        }
        
        print(f"\n{dataset_name} preprocessing completed successfully!")
        print(f"Final dataset shape: {dataset_info['processed_shape']}")
        print(f"Features: {len(dataset_info['features'])}")
        print(f"Classes: {dataset_info['classes']}")
        
        return X_train, X_test, y_train, y_test, dataset_info

def main():
    """
    Test the preprocessing pipeline
    """
    preprocessor = DiabetesDataPreprocessor()
    
    # Test with available datasets
    datasets = ['PIMA', 'DDFH', 'IDPD']
    
    results = {}
    
    for dataset_name in datasets:
        try:
            result = preprocessor.preprocess_dataset(dataset_name, apply_smote_flag=True)
            if result is not None:
                X_train, X_test, y_train, y_test, info = result
                results[dataset_name] = {
                    'data': (X_train, X_test, y_train, y_test),
                    'info': info
                }
                print(f"\n✅ {dataset_name} dataset preprocessing successful")
            else:
                print(f"\n❌ {dataset_name} dataset preprocessing failed")
                
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    
    for dataset_name, result in results.items():
        info = result['info']
        print(f"{dataset_name}:")
        print(f"  - Original shape: {info['original_shape']}")
        print(f"  - Processed shape: {info['processed_shape']}")
        print(f"  - Train/Test: {info['train_size']}/{info['test_size']}")
        print(f"  - Features: {len(info['features'])}")
        print(f"  - Classes: {info['classes']}")

if __name__ == "__main__":
    main()