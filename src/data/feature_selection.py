"""
Feature Selection Module for Ensemble Deep Learning (EDL) Diabetes Prediction System

This module implements feature selection using Extra Tree Classifier (ETC) with Gini importance
as described in the paper: "An Innovative Ensemble Deep Learning Clinical Decision Support System for Diabetes Prediction"

Key functionalities:
1. Use Extra Tree Classifier with Gini importance to rank features
2. Select most relevant features based on importance scores
3. Reduce dimensionality while maintaining model performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ETCFeatureSelector:
    """
    Feature selection class using Extra Tree Classifier with Gini importance
    """
    
    def __init__(self, n_estimators=100, random_state=42):
        """
        Initialize ETC feature selector
        
        Args:
            n_estimators (int): Number of trees in the forest
            random_state (int): Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.etc_model = None
        self.feature_importances_ = None
        self.feature_names_ = None
        self.selected_features_ = None
        
    def fit(self, X, y, feature_names=None):
        """
        Fit Extra Tree Classifier and calculate feature importances
        
        Args:
            X (pd.DataFrame or np.array): Features
            y (pd.Series or np.array): Target
            feature_names (list): List of feature names
        
        Returns:
            self: Fitted selector
        """
        print("Fitting Extra Tree Classifier for feature selection...")
        
        # Initialize Extra Tree Classifier
        self.etc_model = ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            criterion='gini',  # Use Gini importance as specified in paper
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Fit the model
        self.etc_model.fit(X, y)
        
        # Get feature importances
        self.feature_importances_ = self.etc_model.feature_importances_
        
        # Set feature names
        if feature_names is not None:
            self.feature_names_ = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
        
        print(f"ETC fitted successfully with {len(self.feature_names_)} features")
        print(f"Feature importance scores calculated using Gini criterion")
        
        return self
    
    def get_feature_importance_ranking(self, top_k=None):
        """
        Get features ranked by importance
        
        Args:
            top_k (int): Number of top features to return (None for all)
        
        Returns:
            pd.DataFrame: Features ranked by importance
        """
        if self.feature_importances_ is None:
            raise ValueError("ETC model must be fitted first")
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': self.feature_names_,
            'importance': self.feature_importances_
        })
        
        # Sort by importance (descending)
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Add rank
        importance_df['rank'] = range(1, len(importance_df) + 1)
        
        if top_k is not None:
            importance_df = importance_df.head(top_k)
        
        return importance_df
    
    def select_features(self, X, method='top_k', k=None, threshold=None):
        """
        Select features based on importance scores
        
        Args:
            X (pd.DataFrame or np.array): Features
            method (str): Selection method ('top_k', 'threshold', 'auto')
            k (int): Number of top features to select (for 'top_k' method)
            threshold (float): Importance threshold (for 'threshold' method)
        
        Returns:
            pd.DataFrame or np.array: Selected features
        """
        if self.feature_importances_ is None:
            raise ValueError("ETC model must be fitted first")
        
        print(f"Selecting features using method: {method}")
        
        if method == 'top_k':
            if k is None:
                k = max(1, int(len(self.feature_names_) * 0.8))  # Default: 80% of features
                print(f"No k specified, using default: {k} features")
            
            # Get top k features
            importance_ranking = self.get_feature_importance_ranking()
            selected_feature_names = importance_ranking.head(k)['feature'].tolist()
            
        elif method == 'threshold':
            if threshold is None:
                threshold = np.mean(self.feature_importances_)  # Default: above average importance
                print(f"No threshold specified, using mean importance: {threshold:.4f}")
            
            # Select features above threshold
            mask = self.feature_importances_ >= threshold
            selected_feature_names = [name for i, name in enumerate(self.feature_names_) if mask[i]]
            
        elif method == 'auto':
            # Use SelectFromModel for automatic selection
            selector = SelectFromModel(self.etc_model, prefit=True)
            mask = selector.get_support()
            selected_feature_names = [name for i, name in enumerate(self.feature_names_) if mask[i]]
            
        else:
            raise ValueError("Method must be 'top_k', 'threshold', or 'auto'")
        
        self.selected_features_ = selected_feature_names
        
        # Select features from X
        if hasattr(X, 'columns'):  # DataFrame
            X_selected = X[selected_feature_names]
        else:  # numpy array
            feature_indices = [i for i, name in enumerate(self.feature_names_) if name in selected_feature_names]
            X_selected = X[:, feature_indices]
        
        print(f"Selected {len(selected_feature_names)} features out of {len(self.feature_names_)}")
        print(f"Selected features: {selected_feature_names}")
        
        return X_selected
    
    def plot_feature_importance(self, top_k=20, figsize=(10, 8), save_path=None):
        """
        Plot feature importance scores
        
        Args:
            top_k (int): Number of top features to plot
            figsize (tuple): Figure size
            save_path (str): Path to save plot (None to not save)
        """
        if self.feature_importances_ is None:
            raise ValueError("ETC model must be fitted first")
        
        # Get feature importance ranking
        importance_df = self.get_feature_importance_ranking(top_k=top_k)
        
        # Create plot
        plt.figure(figsize=figsize)
        
        # Horizontal bar plot
        y_pos = np.arange(len(importance_df))
        plt.barh(y_pos, importance_df['importance'], color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Customize plot
        plt.yticks(y_pos, importance_df['feature'])
        plt.xlabel('Gini Importance Score')
        plt.ylabel('Features')
        plt.title(f'Top {len(importance_df)} Feature Importance (Extra Tree Classifier)')
        plt.gca().invert_yaxis()  # Highest importance at top
        
        # Add value labels
        for i, (idx, row) in enumerate(importance_df.iterrows()):
            plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def get_summary(self):
        """
        Get summary of feature selection results
        
        Returns:
            dict: Summary information
        """
        if self.feature_importances_ is None:
            raise ValueError("ETC model must be fitted first")
        
        importance_ranking = self.get_feature_importance_ranking()
        
        summary = {
            'total_features': len(self.feature_names_),
            'selected_features': len(self.selected_features_) if self.selected_features_ else 0,
            'top_3_features': importance_ranking.head(3)[['feature', 'importance']].to_dict('records'),
            'mean_importance': np.mean(self.feature_importances_),
            'std_importance': np.std(self.feature_importances_),
            'min_importance': np.min(self.feature_importances_),
            'max_importance': np.max(self.feature_importances_)
        }
        
        return summary

def select_features_for_dataset(X_train, X_test, y_train, dataset_name, 
                               feature_names=None, method='top_k', k=None):
    """
    Apply feature selection to a dataset using ETC
    
    Args:
        X_train: Training features
        X_test: Test features  
        y_train: Training target
        dataset_name (str): Dataset identifier
        feature_names (list): Feature names
        method (str): Selection method
        k (int): Number of features to select
    
    Returns:
        tuple: (X_train_selected, X_test_selected, selector, summary)
    """
    print(f"\n{'='*50}")
    print(f"FEATURE SELECTION FOR {dataset_name} DATASET")
    print(f"{'='*50}")
    
    # Advanced feature selection options
    if method == 'mutual_info':
        from sklearn.feature_selection import mutual_info_classif, SelectKBest
        k_best = k if k is not None else max(1, int(X_train.shape[1] * 0.5))
        print(f"Selecting top {k_best} features using mutual information...")
        selector = SelectKBest(mutual_info_classif, k=k_best)
        selector.fit(X_train, y_train)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        selected_features = [feature_names[i] for i in selector.get_support(indices=True)] if feature_names else selector.get_support(indices=True)
        summary = {
            'total_features': X_train.shape[1],
            'selected_features': len(selected_features),
            'top_3_features': [{'feature': f, 'importance': None} for f in selected_features[:3]],
            'mean_importance': None
        }
        print(f"Selected features: {selected_features}")
        return X_train_selected, X_test_selected, selector, summary
    elif method == 'rfe':
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestClassifier
        k_best = k if k is not None else max(1, int(X_train.shape[1] * 0.5))
        print(f"Selecting top {k_best} features using RFE with RandomForest...")
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=k_best)
        selector.fit(X_train, y_train)
        X_train_selected = selector.transform(X_train)
        X_test_selected = selector.transform(X_test)
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selector.support_[i]] if feature_names else selector.support_
        summary = {
            'total_features': X_train.shape[1],
            'selected_features': len(selected_features),
            'top_3_features': [{'feature': f, 'importance': None} for f in selected_features[:3]],
            'mean_importance': None
        }
        print(f"Selected features: {selected_features}")
        return X_train_selected, X_test_selected, selector, summary
    else:
        # Default ETC feature selection
        selector = ETCFeatureSelector(n_estimators=100, random_state=42)
        selector.fit(X_train, y_train, feature_names=feature_names)
        X_train_selected = selector.select_features(X_train, method=method, k=k)
        X_test_selected = selector.select_features(X_test, method=method, k=k)
        summary = selector.get_summary()
        print(f"\nFeature Selection Summary for {dataset_name}:")
        print(f"- Original features: {summary['total_features']}")
        print(f"- Selected features: {summary['selected_features']}")
        print(f"- Reduction: {((summary['total_features'] - summary['selected_features']) / summary['total_features'] * 100):.1f}%")
        print(f"- Mean importance: {summary['mean_importance']:.4f}")
        print(f"\nTop 3 most important features:")
        for i, feature_info in enumerate(summary['top_3_features']):
            print(f"  {i+1}. {feature_info['feature']}: {feature_info['importance']:.4f}")
        return X_train_selected, X_test_selected, selector, summary

def main():
    """
    Test feature selection with sample data
    """
    from data_preprocessing import DiabetesDataPreprocessor
    
    print("Testing Feature Selection Module")
    print("=" * 50)
    
    # Load and preprocess data
    preprocessor = DiabetesDataPreprocessor()
    
    # Test with PIMA dataset
    try:
        result = preprocessor.preprocess_dataset('PIMA', apply_smote_flag=True)
        if result is not None:
            X_train, X_test, y_train, y_test, info = result
            
            # Apply feature selection
            X_train_selected, X_test_selected, selector, summary = select_features_for_dataset(
                X_train, X_test, y_train, 'PIMA', 
                feature_names=info['features'], 
                method='top_k', k=6
            )
            
            print(f"\nOriginal training shape: {X_train.shape}")
            print(f"Selected training shape: {X_train_selected.shape}")
            
            # Plot feature importance
            selector.plot_feature_importance(top_k=len(info['features']))
            
            print("\n✅ Feature selection test completed successfully!")
            
        else:
            print("❌ Failed to preprocess dataset")
            
    except Exception as e:
        print(f"❌ Error in feature selection test: {str(e)}")

if __name__ == "__main__":
    main()