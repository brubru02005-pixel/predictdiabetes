"""
Advanced Data Preprocessing for Maximum Accuracy Diabetes Prediction

This module implements state-of-the-art preprocessing techniques to maximize model performance:
1. Advanced missing value imputation (KNN, Iterative)
2. Outlier detection and treatment with IQR and Isolation Forest
3. Feature engineering with polynomial and interaction terms
4. Advanced normalization techniques (RobustScaler, QuantileTransform)
5. Sophisticated class balancing with BorderlineSMOTE and ADASYN
6. Clinical feature validation and range checking

Target: Optimize data quality for >99% model accuracy
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTETomek
import scipy.stats as stats
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataPreprocessor:
    """
    Advanced data preprocessing with state-of-the-art techniques
    """
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.feature_engineerers = {}
        self.outlier_detectors = {}
        self.class_balancers = {}
        
        # Clinical validation ranges
        self.clinical_ranges = {
            'glucose': (50, 400),      # mg/dL
            'hba1c': (3.0, 15.0),     # %
            'bmi': (10.0, 70.0),      # kg/m²
            'age': (1, 120),          # years
            'blood_pressure': (40, 250), # mmHg
            'insulin': (0, 1000),     # μU/mL
            'skin_thickness': (0, 100), # mm
            'pregnancies': (0, 20),   # count
            'diabetes_pedigree': (0, 3), # function
            'tg': (30, 1000)          # mg/dL (triglycerides)
        }
    
    def validate_clinical_ranges(self, df, dataset_name):
        """
        Validate data against clinical ranges and flag anomalies
        """
        print(f"\n🏥 Validating clinical ranges for {dataset_name} dataset...")
        
        anomalies_found = 0
        corrections_made = 0
        
        for column in df.columns:
            if column.lower() in self.clinical_ranges or any(key in column.lower() for key in self.clinical_ranges.keys()):
                # Find matching clinical parameter
                param_key = None
                for key in self.clinical_ranges.keys():
                    if key in column.lower() or column.lower() == key:
                        param_key = key
                        break
                
                if param_key:
                    min_val, max_val = self.clinical_ranges[param_key]
                    
                    # Check for values outside clinical ranges
                    outliers = (df[column] < min_val) | (df[column] > max_val)
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        print(f"  ⚠️ {column}: {outlier_count} values outside range [{min_val}, {max_val}]")
                        anomalies_found += outlier_count
                        
                        # Cap extreme values to clinical ranges
                        df[column] = df[column].clip(lower=min_val, upper=max_val)
                        corrections_made += outlier_count
        
        print(f"  📊 Anomalies found: {anomalies_found}")
        print(f"  🔧 Corrections made: {corrections_made}")
        
        return df
    
    def advanced_missing_value_imputation(self, df, strategy='adaptive'):
        """
        Advanced missing value imputation using multiple techniques
        """
        print(f"\n🔍 Advanced missing value imputation using {strategy} strategy...")
        
        # Check missing values
        missing_before = df.isnull().sum().sum()
        print(f"  Missing values before: {missing_before}")
        
        if missing_before == 0:
            print("  ✅ No missing values found")
            return df
        
        # Handle zero values that represent missing data in diabetes datasets
        zero_cols = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
        
        for col in zero_cols:
            if col in df.columns:
                # Replace zero with NaN for proper imputation
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    print(f"  🔄 Converting {zero_count} zero values to NaN in {col}")
                    df[col] = df[col].replace(0, np.nan)
        
        # Choose imputation strategy
        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5, weights='distance')
            df_imputed = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
        elif strategy == 'iterative':
            imputer = IterativeImputer(random_state=42, max_iter=10)
            df_imputed = pd.DataFrame(
                imputer.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
        elif strategy == 'adaptive':
            # Use different strategies for different types of features
            df_imputed = df.copy()
            
            # For glucose, BMI, age - use KNN (important clinical features)
            critical_features = ['glucose', 'bmi', 'age', 'hba1c']
            if any(col in df.columns for col in critical_features):
                critical_cols = [col for col in critical_features if col in df.columns]
                if df[critical_cols].isnull().sum().sum() > 0:
                    knn_imputer = KNNImputer(n_neighbors=3, weights='distance')
                    df_imputed[critical_cols] = knn_imputer.fit_transform(df[critical_cols])
            
            # For other features - use iterative imputation
            remaining_cols = [col for col in df.columns if col not in critical_features]
            if df[remaining_cols].isnull().sum().sum() > 0:
                iter_imputer = IterativeImputer(random_state=42, max_iter=5)
                df_imputed[remaining_cols] = iter_imputer.fit_transform(df[remaining_cols])
        
        missing_after = df_imputed.isnull().sum().sum()
        print(f"  Missing values after: {missing_after}")
        print(f"  ✅ Imputation completed successfully")
        
        return df_imputed
    
    def advanced_outlier_detection(self, df, contamination=0.1):
        """
        Advanced outlier detection using multiple methods
        """
        print(f"\n🎯 Advanced outlier detection (contamination={contamination})...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_detected = 0
        
        # Method 1: IQR method for each feature
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"  📊 {col}: {outlier_count} IQR outliers")
                outliers_detected += outlier_count
                
                # Cap outliers to IQR bounds
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Method 2: Isolation Forest for multivariate outliers
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(df[numeric_cols])
        
        multivariate_outliers = (outlier_labels == -1).sum()
        print(f"  🌲 Isolation Forest: {multivariate_outliers} multivariate outliers")
        
        # Remove extreme multivariate outliers
        df_clean = df[outlier_labels != -1].copy()
        
        print(f"  🔧 Total outliers addressed: {outliers_detected + multivariate_outliers}")
        print(f"  📊 Samples after cleaning: {len(df_clean)} (removed {len(df) - len(df_clean)})")
        
        return df_clean
    
    def advanced_feature_engineering(self, df, dataset_name):
        """
        Advanced feature engineering for maximum predictive power
        """
        print(f"\n⚡ Advanced feature engineering for {dataset_name}...")
        
        original_features = df.shape[1]
        
        # Dataset-specific feature engineering
        if dataset_name == 'PIMA':
            df = self._pima_feature_engineering(df)
        elif dataset_name == 'DDFH':
            df = self._ddfh_feature_engineering(df)
        elif dataset_name == 'COMBINED':
            df = self._combined_feature_engineering(df)
        
        new_features = df.shape[1] - original_features
        print(f"  ✅ Added {new_features} engineered features")
        
        return df
    
    def _pima_feature_engineering(self, df):
        """PIMA-specific feature engineering"""
        
        # BMI categories
        if 'bmi' in df.columns:
            df['bmi_category'] = pd.cut(df['bmi'], 
                                      bins=[0, 18.5, 25, 30, 100], 
                                      labels=[0, 1, 2, 3])  # Underweight, Normal, Overweight, Obese
        
        # Glucose risk levels
        if 'glucose' in df.columns:
            df['glucose_risk'] = pd.cut(df['glucose'],
                                      bins=[0, 100, 125, 200, 500],
                                      labels=[0, 1, 2, 3])  # Normal, Elevated, High, Very High
        
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'],
                                   bins=[0, 30, 45, 60, 120],
                                   labels=[0, 1, 2, 3])  # Young, Middle-aged, Senior, Elderly
        
        # Interaction features
        if 'bmi' in df.columns and 'age' in df.columns:
            df['bmi_age_interaction'] = df['bmi'] * df['age'] / 1000  # Scaled interaction
        
        if 'glucose' in df.columns and 'bmi' in df.columns:
            df['glucose_bmi_ratio'] = df['glucose'] / df['bmi']
        
        if 'insulin' in df.columns and 'glucose' in df.columns:
            df['insulin_glucose_ratio'] = df['insulin'] / (df['glucose'] + 1)  # Avoid division by zero
        
        # Pregnancy risk factor
        if 'pregnancies' in df.columns and 'age' in df.columns:
            df['pregnancy_risk'] = (df['pregnancies'] > 0) & (df['age'] > 35)
            df['pregnancy_risk'] = df['pregnancy_risk'].astype(int)
        
        return df
    
    def _ddfh_feature_engineering(self, df):
        """DDFH-specific feature engineering"""
        
        # HbA1c categories (ADA guidelines)
        if 'hba1c' in df.columns:
            df['hba1c_category'] = pd.cut(df['hba1c'],
                                        bins=[0, 5.7, 6.5, 15],
                                        labels=[0, 1, 2])  # Normal, Pre-diabetes, Diabetes
        
        # BMI-Age risk score
        if 'bmi' in df.columns and 'age' in df.columns:
            df['bmi_age_risk'] = (df['bmi'] / 25) * (df['age'] / 50)  # Normalized risk score
        
        # Gender-specific risk factors
        if 'gender' in df.columns:
            # Encode gender numerically
            df['gender_encoded'] = df['gender'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
            
            if 'age' in df.columns:
                # Gender-age interaction (women have higher risk after menopause)
                df['gender_age_risk'] = np.where(
                    (df['gender_encoded'] == 0) & (df['age'] > 50),
                    1.2,  # Increased risk for post-menopausal women
                    1.0
                )
        
        # Triglycerides risk levels
        if 'tg' in df.columns:
            df['tg_risk'] = pd.cut(df['tg'],
                                 bins=[0, 150, 200, 500, 2000],
                                 labels=[0, 1, 2, 3])  # Normal, Borderline, High, Very High
        
        return df
    
    def _combined_feature_engineering(self, df):
        """Feature engineering for combined dataset"""
        
        # Apply both PIMA and DDFH engineering where applicable
        df = self._pima_feature_engineering(df)
        df = self._ddfh_feature_engineering(df)
        
        # Additional combined features
        if 'glucose' in df.columns and 'hba1c' in df.columns:
            # Glucose-HbA1c correlation feature
            df['glucose_hba1c_consistency'] = np.abs(
                (df['glucose'] - 100) / 100 - (df['hba1c'] - 5.7) / 5.7
            )
        
        return df
    
    def advanced_normalization(self, X_train, X_test, method='robust'):
        """
        Advanced normalization techniques
        """
        print(f"\n📏 Advanced normalization using {method} method...")
        
        if method == 'robust':
            scaler = RobustScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"  ✅ Normalization completed using {scaler.__class__.__name__}")
        
        return X_train_scaled, X_test_scaled, scaler
    
    def advanced_class_balancing(self, X, y, strategy='adaptive'):
        """
        Advanced class balancing techniques
        """
        print(f"\n⚖️ Advanced class balancing using {strategy} strategy...")
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"  Original class distribution: {dict(zip(unique, counts))}")
        
        if strategy == 'borderline_smote':
            balancer = BorderlineSMOTE(random_state=42, k_neighbors=3)
        elif strategy == 'adasyn':
            balancer = ADASYN(random_state=42, n_neighbors=3)
        elif strategy == 'smote_tomek':
            balancer = SMOTETomek(random_state=42)
        elif strategy == 'adaptive':
            # Choose best strategy based on data characteristics
            if len(unique) == 2:
                imbalance_ratio = max(counts) / min(counts)
                if imbalance_ratio > 5:
                    balancer = BorderlineSMOTE(random_state=42, k_neighbors=3)
                    print("  📊 High imbalance detected, using BorderlineSMOTE")
                else:
                    balancer = SMOTE(random_state=42, k_neighbors=3)
                    print("  📊 Moderate imbalance, using standard SMOTE")
            else:
                balancer = ADASYN(random_state=42, n_neighbors=3)
                print("  📊 Multi-class detected, using ADASYN")
        else:
            balancer = SMOTE(random_state=42, k_neighbors=3)
        
        try:
            X_balanced, y_balanced = balancer.fit_resample(X, y)
            
            # Check new distribution
            unique_new, counts_new = np.unique(y_balanced, return_counts=True)
            print(f"  Balanced class distribution: {dict(zip(unique_new, counts_new))}")
            print(f"  ✅ Class balancing completed: {len(X)} -> {len(X_balanced)} samples")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            print(f"  ⚠️ Class balancing failed: {e}")
            print("  🔄 Falling back to original data")
            return X, y
    
    def create_polynomial_features(self, X_train, X_test, degree=2, interaction_only=False):
        """
        Create polynomial and interaction features
        """
        print(f"\n🧮 Creating polynomial features (degree={degree}, interaction_only={interaction_only})...")
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        
        # Fit on training data
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        
        feature_names = poly.get_feature_names_out([f'f{i}' for i in range(X_train.shape[1])])
        
        original_features = X_train.shape[1]
        new_features = X_train_poly.shape[1] - original_features
        
        print(f"  📊 Original features: {original_features}")
        print(f"  ✨ New features: {new_features}")
        print(f"  📈 Total features: {X_train_poly.shape[1]}")
        
        return X_train_poly, X_test_poly, poly, feature_names
    
    def advanced_feature_selection(self, X_train, X_test, y_train, method='mutual_info_enhanced', k=None):
        """
        Advanced feature selection with multiple criteria
        """
        print(f"\n🎯 Advanced feature selection using {method}...")
        
        if k is None:
            k = min(20, max(6, X_train.shape[1] // 3))  # Adaptive k selection
        
        if method == 'mutual_info_enhanced':
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            
            # Enhanced mutual information with normalization
            selector = SelectKBest(
                score_func=lambda X, y: mutual_info_classif(X, y, random_state=42),
                k=k
            )
            
        elif method == 'chi2_enhanced':
            from sklearn.feature_selection import SelectKBest, chi2
            from sklearn.preprocessing import MinMaxScaler
            
            # Ensure non-negative values for chi2
            scaler = MinMaxScaler()
            X_train_pos = scaler.fit_transform(X_train)
            X_test_pos = scaler.transform(X_test)
            
            selector = SelectKBest(score_func=chi2, k=k)
            X_train_selected = selector.fit_transform(X_train_pos, y_train)
            X_test_selected = selector.transform(X_test_pos)
            
            scores = selector.scores_
            selected_indices = selector.get_support(indices=True)
            
            print(f"  ✅ Selected {k} features using Chi2")
            print(f"  📊 Feature scores range: {scores.min():.3f} - {scores.max():.3f}")
            
            return X_train_selected, X_test_selected, selector, {
                'method': method,
                'selected_features': selected_indices,
                'scores': scores
            }
        
        elif method == 'rfe_enhanced':
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestClassifier
            
            # Use Random Forest for recursive feature elimination
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(estimator=estimator, n_features_to_select=k, step=1)
        
        elif method == 'variance_threshold_enhanced':
            from sklearn.feature_selection import VarianceThreshold
            
            # Remove low-variance features first
            var_threshold = VarianceThreshold(threshold=0.01)
            X_train_var = var_threshold.fit_transform(X_train)
            X_test_var = var_threshold.transform(X_test)
            
            # Then apply mutual information
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            X_train_selected = selector.fit_transform(X_train_var, y_train)
            X_test_selected = selector.transform(X_test_var)
            
            print(f"  🔍 Variance threshold + Mutual Info selection")
            print(f"  ✅ Selected {k} features from {X_train_var.shape[1]} variance-filtered features")
            
            return X_train_selected, X_test_selected, selector, {
                'method': method,
                'variance_threshold': var_threshold,
                'feature_selector': selector
            }
        
        # Default execution for standard methods
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        scores = getattr(selector, 'scores_', None)
        selected_indices = selector.get_support(indices=True)
        
        print(f"  ✅ Selected {k} features using {method}")
        if scores is not None:
            print(f"  📊 Feature scores range: {scores.min():.3f} - {scores.max():.3f}")
        
        return X_train_selected, X_test_selected, selector, {
            'method': method,
            'selected_features': selected_indices,
            'scores': scores
        }
    
    def preprocess_dataset_advanced(self, dataset_name, test_size=0.2, random_state=42, 
                                  apply_balancing=True, feature_engineering=True):
        """
        Complete advanced preprocessing pipeline for maximum accuracy
        """
        print(f"\n{'='*80}")
        print(f"🚀 ADVANCED PREPROCESSING FOR {dataset_name} DATASET")
        print(f"Target: Maximum data quality for >99% model accuracy")
        print(f"{'='*80}")
        
        try:
            # Load dataset
            df = self.load_dataset(dataset_name)
            print(f"📊 Loaded dataset: {df.shape}")
            
            # Step 1: Clinical validation
            df = self.validate_clinical_ranges(df, dataset_name)
            
            # Step 2: Missing value imputation
            df = self.advanced_missing_value_imputation(df, strategy='adaptive')
            
            # Step 3: Outlier detection and treatment
            df = self.advanced_outlier_detection(df, contamination=0.05)
            
            # Step 4: Advanced feature engineering
            if feature_engineering:
                df = self.advanced_feature_engineering(df, dataset_name)
            
            # Step 5: Prepare features and target
            target_col = df.columns[-1]  # Assume last column is target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Step 6: Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y
            )
            
            print(f"📊 Train-test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
            
            # Step 7: Advanced normalization
            X_train_scaled, X_test_scaled, scaler = self.advanced_normalization(
                X_train, X_test, method='robust'
            )
            
            # Step 8: Advanced class balancing
            if apply_balancing:
                X_train_balanced, y_train_balanced = self.advanced_class_balancing(
                    X_train_scaled, y_train, strategy='adaptive'
                )
            else:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
            
            # Step 9: Feature selection for optimal performance
            X_train_final, X_test_final, selector, selection_info = self.advanced_feature_selection(
                X_train_balanced, X_test_scaled, y_train_balanced, 
                method='mutual_info_enhanced', k=min(15, X_train_balanced.shape[1])
            )
            
            # Store preprocessing artifacts
            self.scalers[dataset_name] = scaler
            
            # Final statistics
            print(f"\n📈 PREPROCESSING SUMMARY:")
            print(f"  Original shape: {df.shape}")
            print(f"  Final train shape: {X_train_final.shape}")
            print(f"  Final test shape: {X_test_final.shape}")
            print(f"  Classes: {len(np.unique(y))}")
            print(f"  Features selected: {X_train_final.shape[1]}")
            
            info = {
                'dataset_name': dataset_name,
                'original_shape': df.shape,
                'final_shape': (X_train_final.shape[0] + X_test_final.shape[0], X_train_final.shape[1]),
                'num_classes': len(np.unique(y)),
                'features': X.columns.tolist(),
                'target': target_col,
                'preprocessing_steps': [
                    'Clinical validation',
                    'Missing value imputation',
                    'Outlier detection',
                    'Feature engineering',
                    'Advanced normalization',
                    'Class balancing',
                    'Feature selection'
                ]
            }
            
            print(f"✅ Advanced preprocessing completed successfully!")
            
            return X_train_final, X_test_final, y_train_balanced, y_test, info
            
        except Exception as e:
            print(f"❌ Error in advanced preprocessing: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_dataset(self, dataset_name):
        """Load dataset with enhanced error handling"""
        
        dataset_paths = {
            'PIMA': ['data/raw/pima_dataset.csv', 'data/diabetes.csv', 'data/pima.csv.csv'],
            'DDFH': ['data/raw/ddfh_dataset.csv', 'data/Dataset of Diabetes .csv'],
            'COMBINED': ['data/processed/combined_diabetes.csv', 'data/combined_diabetes.csv']
        }
        
        if dataset_name not in dataset_paths:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Try multiple possible paths
        for path in dataset_paths[dataset_name]:
            if os.path.exists(path):
                print(f"📂 Loading {dataset_name} from {path}")
                df = pd.read_csv(path)
                print(f"  📊 Shape: {df.shape}")
                return df
        
        raise FileNotFoundError(f"No valid file found for {dataset_name} dataset")

def main():
    """Test advanced preprocessing"""
    print("🧪 Testing Advanced Data Preprocessing...")
    
    preprocessor = AdvancedDataPreprocessor()
    
    # Test with available datasets
    datasets_to_test = ['PIMA', 'DDFH']
    
    for dataset_name in datasets_to_test:
        try:
            print(f"\n{'='*60}")
            print(f"Testing {dataset_name} Dataset")
            print(f"{'='*60}")
            
            result = preprocessor.preprocess_dataset_advanced(
                dataset_name, 
                apply_balancing=True,
                feature_engineering=True
            )
            
            if result is not None:
                X_train, X_test, y_train, y_test, info = result
                print(f"✅ {dataset_name} preprocessing successful!")
                print(f"  📊 Final training shape: {X_train.shape}")
                print(f"  🎯 Ready for enhanced model training")
            else:
                print(f"❌ {dataset_name} preprocessing failed")
                
        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {str(e)}")
    
    print("\n✅ Advanced preprocessing testing completed!")

if __name__ == "__main__":
    main()
