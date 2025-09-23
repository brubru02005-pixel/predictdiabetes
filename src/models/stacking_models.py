"""
Meta-Level Stacking Models for Ensemble Deep Learning (EDL) Diabetes Prediction System

This module implements the three meta-level stacking models as described in the paper:
"An Innovative Ensemble Deep Learning Clinical Decision Support System for Diabetes Prediction"

Meta-models implemented:
1. Stack-ANN: Neural network that takes base model predictions as input
2. Stack-LSTM: LSTM network processing base model outputs
3. Stack-CNN: CNN processing base model predictions

These models use the predictions from ANN, LSTM, and CNN base models as input features
and learn to make final predictions by combining the base model outputs.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models, callbacks
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class StackingModelBuilder:
    def grid_search_stack_lr(self, X_train, y_train, X_test, y_test):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        print("Grid search for LogisticRegression meta-model...")
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
        grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=self.random_state), param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)
        print(f"Best LR params: {grid.best_params_}, best CV score: {grid.best_score_:.4f}")
        best_model = grid.best_estimator_
        acc = best_model.score(X_test, y_test)
        print(f"Test accuracy with best LR: {acc:.4f}")
        return best_model, grid.best_params_, acc

    def grid_search_stack_rf(self, X_train, y_train, X_test, y_test):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        print("Grid search for RandomForest meta-model...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        grid = GridSearchCV(RandomForestClassifier(random_state=self.random_state), param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)
        print(f"Best RF params: {grid.best_params_}, best CV score: {grid.best_score_:.4f}")
        best_model = grid.best_estimator_
        acc = best_model.score(X_test, y_test)
        print(f"Test accuracy with best RF: {acc:.4f}")
        return best_model, grid.best_params_, acc

    def grid_search_stack_xgb(self, X_train, y_train, X_test, y_test):
        from xgboost import XGBClassifier
        from sklearn.model_selection import GridSearchCV
        print("Grid search for XGBoost meta-model...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.random_state), param_grid, cv=3, scoring='accuracy')
        grid.fit(X_train, y_train)
        print(f"Best XGB params: {grid.best_params_}, best CV score: {grid.best_score_:.4f}")
        best_model = grid.best_estimator_
        acc = best_model.score(X_test, y_test)
        print(f"Test accuracy with best XGB: {acc:.4f}")
        return best_model, grid.best_params_, acc
    def build_stack_rf_model(self, input_dim):
        """
        Build RandomForest meta-model for stacking
        Args:
            input_dim (int): Number of input features
        Returns:
            RandomForestClassifier: Fitted model
        """
        from sklearn.ensemble import RandomForestClassifier
        print("Building RandomForest meta-model for stacking...")
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        return model

    def build_stack_lr_model(self, input_dim):
        """
        Build LogisticRegression meta-model for stacking
        Args:
            input_dim (int): Number of input features
        Returns:
            LogisticRegression: Fitted model
        """
        from sklearn.linear_model import LogisticRegression
        print("Building LogisticRegression meta-model for stacking...")
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        return model

    def build_stack_xgb_model(self, input_dim):
        """
        Build XGBoost meta-model for stacking
        Args:
            input_dim (int): Number of input features
        Returns:
            XGBClassifier: Fitted model
        """
        from xgboost import XGBClassifier
        print("Building XGBoost meta-model for stacking...")
        model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=self.random_state)
        return model

    def train_sklearn_stacking_model(self, model, X_train, y_train, model_name):
        print(f"\nTraining {model_name} stacking model (sklearn)...")
        model.fit(X_train, y_train)
        print(f"{model_name} training completed!")
        return model

    def evaluate_sklearn_stacking_model(self, model, X_test, y_test, model_name):
        from sklearn.metrics import accuracy_score, classification_report
        print(f"\nEvaluating {model_name} stacking model (sklearn)...")
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)
        else:
            y_pred_proba = None
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} Test Accuracy: {accuracy:.4f}")
        report = classification_report(y_test, y_pred, output_dict=True)
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report
        }
        return results

    def save_sklearn_stacking_model(self, model, model_name, dataset_name):
        import joblib
        model_path = f'models/{model_name}_{dataset_name}.joblib'
        joblib.dump(model, model_path)
        print(f"{model_name} stacking model saved to: {model_path}")
    """
    Builder class for meta-level stacking models
    """
    
    def __init__(self, num_base_models=3, num_classes=2, random_state=42):
        """
        Initialize stacking model builder
        
        Args:
            num_base_models (int): Number of base models (default: 3 for ANN, LSTM, CNN)
            num_classes (int): Number of classes (2 for binary, >2 for multi-class)
            random_state (int): Random state for reproducibility
        """
        self.num_base_models = num_base_models
        self.num_classes = num_classes
        self.random_state = random_state
        # Improved hyperparameters for higher accuracy
        self.epochs = 120  # More epochs for deeper training
        self.batch_size = 8  # Even smaller batch size for better generalization
        self.learning_rate = 0.001  # Lower learning rate for finer convergence
        self.validation_split = 0.2
        # Model storage
        self.models = {}
        self.histories = {}
    
    def prepare_stacking_data(self, base_predictions_dict, targets):
        """
        Prepare input data for stacking models from base model predictions
        
        Args:
            base_predictions_dict (dict): Dictionary containing predictions from base models
                Format: {'ANN': predictions, 'LSTM': predictions, 'CNN': predictions}
            targets: Target values
        
        Returns:
            tuple: (stacking_features, targets)
        """
        print("Preparing stacking data from base model predictions...")
        
        # Collect predictions from all base models
        predictions_list = []
        model_names = []
        
        for model_name, predictions in base_predictions_dict.items():
            if predictions is not None and len(predictions) > 0:
                # Ensure predictions are in the right format
                if len(predictions.shape) == 1:
                    predictions = predictions.reshape(-1, 1)
                elif len(predictions.shape) == 2 and predictions.shape[1] > 1:
                    # For multi-class, use probabilities as features
                    pass
                else:
                    predictions = predictions.reshape(-1, 1)
                
                predictions_list.append(predictions)
                model_names.append(model_name)
        
        if len(predictions_list) == 0:
            raise ValueError("No base model predictions available for stacking")
        
        # Stack predictions horizontally to create feature matrix
        stacking_features = np.concatenate(predictions_list, axis=1)
        
        print(f"Stacking data prepared:")
        print(f"- Base models used: {model_names}")
        print(f"- Stacking features shape: {stacking_features.shape}")
        print(f"- Features per model: {[pred.shape[1] if len(pred.shape) > 1 else 1 for pred in predictions_list]}")
        
        return stacking_features, targets
    
    def build_stack_ann_model(self, input_dim):
        """
        Build Stack-ANN (Artificial Neural Network) stacking model
        
        Args:
            input_dim (int): Number of input features (base model predictions)
        
        Returns:
            keras.Model: Compiled Stack-ANN model
        """
        print("Building Stack-ANN (Stacking Artificial Neural Network) model...")
        
        # Enhanced Stack-ANN with more layers, units, and regularization
        if self.num_classes > 2:
            model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            loss = 'sparse_categorical_crossentropy'
        else:
            model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            loss = 'binary_crossentropy'
        optimizer = Adam(learning_rate=self.learning_rate)
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"Stack-ANN model built successfully:")
        print(f"- Input dimension: {input_dim}")
        print(f"- Output classes: {self.num_classes}")
        print(f"- Architecture: {input_dim} -> 128 -> 64 -> 32 -> 16 -> {'softmax' if self.num_classes > 2 else 'sigmoid'}")
        return model
    
    def build_stack_lstm_model(self, input_dim):
        """
        Build Stack-LSTM (Long Short-Term Memory) stacking model
        
        Args:
            input_dim (int): Number of input features (base model predictions)
        
        Returns:
            keras.Model: Compiled Stack-LSTM model
        """
        print("Building Stack-LSTM (Stacking Long Short-Term Memory) model...")
        
        # More LSTM units, layers, and regularization
        if self.num_classes > 2:
            model = models.Sequential([
                layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
                layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
                layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
                layers.LSTM(16, dropout=0.2, recurrent_dropout=0.1),
                layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(0.2),
                layers.Dense(8, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            loss = 'sparse_categorical_crossentropy'
        else:
            model = models.Sequential([
                layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
                layers.LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
                layers.LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.1),
                layers.LSTM(16, dropout=0.2, recurrent_dropout=0.1),
                layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(0.2),
                layers.Dense(8, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            loss = 'binary_crossentropy'
        optimizer = Adam(learning_rate=self.learning_rate)
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"Stack-LSTM model built successfully:")
        print(f"- Input dimension: {input_dim}")
        print(f"- Output classes: {self.num_classes}")
        print(f"- Architecture: LSTM(32) -> LSTM(16) -> Dense(8) -> {'softmax' if self.num_classes > 2 else 'sigmoid'}")
        return model
    
    def build_stack_cnn_model(self, input_dim):
        """
        Build Stack-CNN (Convolutional Neural Network) stacking model
        
        Args:
            input_dim (int): Number of input features (base model predictions)
        
        Returns:
            keras.Model: Compiled Stack-CNN model
        """
        print("Building Stack-CNN (Stacking Convolutional Neural Network) model...")
        
        # More filters, layers, and regularization for CNN meta-model
        if self.num_classes > 2:
            model = models.Sequential([
                layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
                layers.Conv1D(32, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.GlobalMaxPooling1D(),
                layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(0.2),
                layers.Dense(8, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            loss = 'sparse_categorical_crossentropy'
        else:
            model = models.Sequential([
                layers.Reshape((input_dim, 1), input_shape=(input_dim,)),
                layers.Conv1D(32, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.GlobalMaxPooling1D(),
                layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(0.3),
                layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
                layers.Dropout(0.2),
                layers.Dense(8, activation='relu'),
                layers.Dense(1, activation='sigmoid')
            ])
            loss = 'binary_crossentropy'
        optimizer = Adam(learning_rate=self.learning_rate)
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"Stack-CNN model built successfully:")
        print(f"- Input dimension: {input_dim}")
        print(f"- Output classes: {self.num_classes}")
        print(f"- Architecture: Conv1D(16) -> Conv1D(32) -> GlobalMaxPool -> Dense(16) -> Dense(8) -> {'softmax' if self.num_classes > 2 else 'sigmoid'}")
        return model
    
    def train_stacking_model(self, model, X_train, y_train, model_name, verbose=1):
        """
        Train a stacking model
        
        Args:
            model: Keras model to train
            X_train: Training features (base model predictions)
            y_train: Training targets
            model_name (str): Name of the stacking model
            verbose (int): Verbosity level
        
        Returns:
            tuple: (trained_model, training_history)
        """
        print(f"\nTraining {model_name} stacking model...")
        print(f"Training data shape: {X_train.shape}")
        print(f"Training parameters: epochs={self.epochs}, batch_size={self.batch_size}")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2, 
            patience=3, 
            min_lr=0.001
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        print(f"{model_name} training completed!")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        return model, history
    
    def evaluate_stacking_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a trained stacking model
        
        Args:
            model: Trained Keras model
            X_test: Test features (base model predictions)
            y_test: Test targets
            model_name (str): Name of the stacking model
        
        Returns:
            dict: Evaluation results
        """
        print(f"\nEvaluating {model_name} stacking model...")
        
        # Get predictions
        if self.num_classes > 2:
            y_pred_proba = model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred_proba = model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{model_name} Test Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report
        }
        
        return results
    
    def save_stacking_model(self, model, model_name, dataset_name):
        """
        Save trained stacking model
        
        Args:
            model: Trained Keras model
            model_name (str): Name of the stacking model
            dataset_name (str): Dataset identifier
        """
        model_path = f'models/{model_name}_{dataset_name}.h5'
        model.save(model_path)
        print(f"{model_name} stacking model saved to: {model_path}")
    
    def build_and_train_all_stacking_models(self, X_train_stack, y_train, X_test_stack, y_test, dataset_name):
        """
        Build and train all three stacking models
        
        Args:
            X_train_stack: Training features (base model predictions)
            y_train: Training targets
            X_test_stack: Test features (base model predictions)
            y_test: Test targets
            dataset_name (str): Dataset identifier
        
        Returns:
            dict: Results for all stacking models
        """
        print(f"\n{'='*60}")
        print(f"TRAINING META-LEVEL STACKING MODELS FOR {dataset_name} DATASET")
        print(f"{'='*60}")
        
        results = {}
        input_dim = X_train_stack.shape[1]
        
        model_builders = {
            'Stack-ANN': (self.build_stack_ann_model, 'keras'),
            'Stack-LSTM': (self.build_stack_lstm_model, 'keras'),
            'Stack-CNN': (self.build_stack_cnn_model, 'keras'),
            'Stack-RF': (self.build_stack_rf_model, 'sklearn'),
            'Stack-LR': (self.build_stack_lr_model, 'sklearn'),
            'Stack-XGB': (self.build_stack_xgb_model, 'sklearn'),
        }

        for model_name, (builder_func, model_type) in model_builders.items():
            try:
                print(f"\n{'-'*40}")
                print(f"Processing {model_name} Model")
                print(f"{'-'*40}")
                model = builder_func(input_dim)
                if model_type == 'keras':
                    trained_model, history = self.train_stacking_model(
                        model, X_train_stack, y_train, model_name
                    )
                    evaluation = self.evaluate_stacking_model(
                        trained_model, X_test_stack, y_test, model_name
                    )
                    self.save_stacking_model(trained_model, model_name, dataset_name)
                else:
                    trained_model = self.train_sklearn_stacking_model(
                        model, X_train_stack, y_train, model_name
                    )
                    history = None
                    evaluation = self.evaluate_sklearn_stacking_model(
                        trained_model, X_test_stack, y_test, model_name
                    )
                    self.save_sklearn_stacking_model(trained_model, model_name, dataset_name)
                results[model_name] = {
                    'model': trained_model,
                    'history': history,
                    'evaluation': evaluation,
                    'status': 'success'
                }
                print(f"✅ {model_name} model completed successfully!")
            except Exception as e:
                print(f"❌ Error training {model_name} model: {str(e)}")
                results[model_name] = {'error': str(e), 'status': 'failed'}
        return results

def train_stacking_models_for_dataset(base_predictions_train, base_predictions_test, 
                                    y_train, y_test, dataset_name, num_classes=2):
    """
    Train all stacking models for a specific dataset
    
    Args:
        base_predictions_train (dict): Training predictions from base models
        base_predictions_test (dict): Test predictions from base models
        y_train: Training targets
        y_test: Test targets
        dataset_name (str): Dataset identifier
        num_classes (int): Number of classes
    
    Returns:
        tuple: (results, final_predictions)
    """
    print(f"\nTraining meta-level stacking models for {dataset_name} dataset...")
    
    # Initialize stacking model builder
    builder = StackingModelBuilder(
        num_base_models=len(base_predictions_train),
        num_classes=num_classes,
        random_state=42
    )
    
    # Prepare stacking data
    X_train_stack, y_train_stack = builder.prepare_stacking_data(base_predictions_train, y_train)
    X_test_stack, y_test_stack = builder.prepare_stacking_data(base_predictions_test, y_test)
    
    print(f"Stacking training data shape: {X_train_stack.shape}")
    print(f"Stacking test data shape: {X_test_stack.shape}")
    
    # Train all stacking models (default hyperparameters)
    results = builder.build_and_train_all_stacking_models(
        X_train_stack, y_train_stack, X_test_stack, y_test_stack, dataset_name
    )

    # Collect final predictions from stacking models
    final_predictions = {}
    for model_name, model_results in results.items():
        if 'error' not in model_results:
            model = model_results['model']
            if num_classes > 2:
                train_pred = model.predict(X_train_stack)
                test_pred = model.predict(X_test_stack)
            else:
                train_pred = model.predict(X_train_stack).flatten()
                test_pred = model.predict(X_test_stack).flatten()
            final_predictions[model_name] = {
                'train_predictions': train_pred,
                'test_predictions': test_pred
            }

    # Grid search for best classic meta-models
    print("\n===== GRID SEARCH FOR STACKING META-MODELS (LR, RF, XGB) =====")
    best_lr, best_lr_params, best_lr_acc = builder.grid_search_stack_lr(X_train_stack, y_train_stack, X_test_stack, y_test_stack)
    best_rf, best_rf_params, best_rf_acc = builder.grid_search_stack_rf(X_train_stack, y_train_stack, X_test_stack, y_test_stack)
    best_xgb, best_xgb_params, best_xgb_acc = builder.grid_search_stack_xgb(X_train_stack, y_train_stack, X_test_stack, y_test_stack)

    print(f"\nBest tuned LR stacking accuracy: {best_lr_acc:.4f} | Params: {best_lr_params}")
    print(f"Best tuned RF stacking accuracy: {best_rf_acc:.4f} | Params: {best_rf_params}")
    print(f"Best tuned XGB stacking accuracy: {best_xgb_acc:.4f} | Params: {best_xgb_params}")

    # Save best tuned models
    import joblib
    joblib.dump(best_lr, f"models/Stack-LR-TUNED_{dataset_name}.joblib")
    joblib.dump(best_rf, f"models/Stack-RF-TUNED_{dataset_name}.joblib")
    joblib.dump(best_xgb, f"models/Stack-XGB-TUNED_{dataset_name}.joblib")

    print("\nTuned stacking meta-models saved.")

    return results, final_predictions

def main():
    """
    Test stacking models training
    """
    from data_preprocessing import DiabetesDataPreprocessor
    from feature_selection import select_features_for_dataset
    from base_models import train_base_models_for_dataset
    
    print("Testing Meta-Level Stacking Models Training")
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
            
            # Train base models
            base_results, base_predictions = train_base_models_for_dataset(
                X_train_selected, X_test_selected, y_train, y_test, 
                'PIMA', num_classes=len(info['classes'])
            )
            
            if len(base_predictions) > 0:
                # Extract train and test predictions for stacking
                base_train_preds = {name: preds['train_predictions'] 
                                  for name, preds in base_predictions.items()}
                base_test_preds = {name: preds['test_predictions'] 
                                 for name, preds in base_predictions.items()}
                
                # Train stacking models
                stack_results, final_predictions = train_stacking_models_for_dataset(
                    base_train_preds, base_test_preds, y_train, y_test,
                    'PIMA', num_classes=len(info['classes'])
                )
                
                print("\n✅ Complete EDL system test completed successfully!")
                
                # Compare base vs stacking model performance
                print("\n" + "="*60)
                print("PERFORMANCE COMPARISON: BASE vs STACKING MODELS")
                print("="*60)
                
                print("\nBase Models:")
                for model_name, results in base_results.items():
                    if 'error' not in results:
                        accuracy = results['evaluation']['accuracy']
                        print(f"  {model_name:4s}: {accuracy:.4f}")
                
                print("\nStacking Models:")
                for model_name, results in stack_results.items():
                    if 'error' not in results:
                        accuracy = results['evaluation']['accuracy']
                        print(f"  {model_name:10s}: {accuracy:.4f}")
                
            else:
                print("❌ No base model predictions available for stacking")
            
        else:
            print("❌ Failed to preprocess dataset")
            
    except Exception as e:
        print(f"❌ Error in stacking models training test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()