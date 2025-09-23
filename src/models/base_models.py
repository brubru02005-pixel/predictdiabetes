"""
Base-Level Models for Ensemble Deep Learning (EDL) Diabetes Prediction System

This module implements the three base-level deep learning models as described in the paper:
"An Innovative Ensemble Deep Learning Clinical Decision Support System for Diabetes Prediction"

Base models implemented:
1. ANN (Artificial Neural Network): Multi-layer feedforward neural network
2. LSTM (Long Short-Term Memory): Recurrent neural network for sequence modeling
3. CNN (Convolutional Neural Network): 1D CNN for feature pattern recognition

These models are trained independently and their predictions are used as input 
for the meta-level stacking models.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models, callbacks
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BaseModelBuilder:
    """
    Builder class for base-level deep learning models
    """
    
    def __init__(self, input_dim, num_classes=2, random_state=42):
        """
        Initialize base model builder
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of classes (2 for binary, >2 for multi-class)
            random_state (int): Random state for reproducibility
        """
        self.input_dim = input_dim
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
        
    def build_ann_model(self):
        """
        Build ANN (Artificial Neural Network) model
        
        Returns:
            keras.Model: Compiled ANN model
        """
        print("Building ANN (Artificial Neural Network) model...")
        
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1)
        ])
        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        loss = 'sparse_categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f"ANN model built successfully:")
        print(f"- Input dimension: {self.input_dim}")
        print(f"- Output classes: {self.num_classes}")
        if self.num_classes == 2:
            print(f"- Architecture: {self.input_dim} -> 64 -> 32 -> 16 -> 1 neuron (sigmoid)")
        else:
            print(f"- Architecture: {self.input_dim} -> 64 -> 32 -> 16 -> {self.num_classes} neurons (softmax)")
        
        return model
    
    def build_lstm_model(self):
        """
        Build LSTM (Long Short-Term Memory) model
        
        Returns:
            keras.Model: Compiled LSTM model
        """
        print("Building LSTM (Long Short-Term Memory) model...")
        
        model = models.Sequential([
            layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
            layers.LSTM(64, return_sequences=True, dropout=0.3),
            layers.LSTM(32, return_sequences=True, dropout=0.2),
            layers.LSTM(16, dropout=0.2),
            layers.Dense(8, activation='relu'),
            layers.Dropout(0.1)
        ])
        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        loss = 'sparse_categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy']
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f"LSTM model built successfully:")
        print(f"- Input dimension: {self.input_dim}")
        print(f"- Output classes: {self.num_classes}")
        if self.num_classes == 2:
            print(f"- Architecture: LSTM(64) -> LSTM(32) -> LSTM(16) -> Dense(8) -> 1 neuron (sigmoid)")
        else:
            print(f"- Architecture: LSTM(64) -> LSTM(32) -> LSTM(16) -> Dense(8) -> {self.num_classes} neurons (softmax)")
        
        return model
    
    def build_cnn_model(self):
        """
        Build CNN (Convolutional Neural Network) model
        
        Returns:
            keras.Model: Compiled CNN model
        """
        print("Building CNN (Convolutional Neural Network) model...")
        
        # Enhanced 1D CNN for tabular data
        model = models.Sequential([
            layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2)
        ])
        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        loss = 'sparse_categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy'
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"CNN model built successfully:")
        print(f"- Input dimension: {self.input_dim}")
        print(f"- Output classes: {self.num_classes}")
        if self.num_classes == 2:
            print(f"- Architecture: Conv1D(64) -> Conv1D(128) -> Conv1D(64) -> Dense(64) -> Dense(32) -> Dense(16) -> 1 neuron (sigmoid)")
        else:
            print(f"- Architecture: Conv1D(64) -> Conv1D(128) -> Conv1D(64) -> Dense(64) -> Dense(32) -> Dense(16) -> {self.num_classes} neurons (softmax)")
        
        return model
    
    def train_model(self, model, model_name, X_train, X_test, y_train, y_test):
        """
        Train a deep learning model
        
        Args:
            model: Keras model to train
            model_name (str): Name of the model
            X_train: Training features
            X_test: Testing features
            y_train: Training targets
            y_test: Testing targets
        
        Returns:
            dict: Training results including model, history, and evaluation
        """
        print(f"\nTraining {model_name} model...")
        
        try:
            # Early stopping callback
            early_stopping = callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            )
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Make predictions
            train_pred_prob = model.predict(X_train, verbose=0)
            test_pred_prob = model.predict(X_test, verbose=0)
            
            if self.num_classes == 2:
                train_pred = (train_pred_prob > 0.5).astype(int)
                test_pred = (test_pred_prob > 0.5).astype(int)
            else:
                train_pred = np.argmax(train_pred_prob, axis=1)
                test_pred = np.argmax(test_pred_prob, axis=1)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Additional metrics
            precision = precision_score(y_test, test_pred, average='weighted')
            recall = recall_score(y_test, test_pred, average='weighted')
            f1 = f1_score(y_test, test_pred, average='weighted')
            
            # ROC AUC (handle multi-class)
            if self.num_classes == 2:
                roc_auc = roc_auc_score(y_test, test_pred_prob)
            else:
                try:
                    roc_auc = roc_auc_score(y_test, test_pred_prob, multi_class='ovr')
                except:
                    roc_auc = 0.0  # Default if calculation fails
            
            # Matthews Correlation Coefficient
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(y_test, test_pred)
            
            # Specificity calculation
            cm = confusion_matrix(y_test, test_pred)
            if self.num_classes == 2:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                # For multi-class, calculate average specificity
                specificity = 0.0
                for i in range(self.num_classes):
                    tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
                    fp = np.sum(cm[:, i]) - cm[i, i]
                    if (tn + fp) > 0:
                        specificity += tn / (tn + fp)
                specificity /= self.num_classes
            
            # Store results
            results = {
                'model': model,
                'history': history,
                'evaluation': {
                    'accuracy': test_accuracy,
                    'train_accuracy': train_accuracy,
                    'precision': precision,
                    'sensitivity': recall,  # Recall is same as sensitivity
                    'specificity': specificity,
                    'f1_score': f1,
                    'mcc': mcc,
                    'roc_auc': roc_auc
                },
                'predictions': {
                    'train_predictions': train_pred,
                    'test_predictions': test_pred,
                    'train_probabilities': train_pred_prob,
                    'test_probabilities': test_pred_prob
                },
                'status': 'success'
            }
            
            print(f"✓ {model_name} training completed successfully")
            print(f"  - Train Accuracy: {train_accuracy:.4f}")
            print(f"  - Test Accuracy: {test_accuracy:.4f}")
            print(f"  - Precision: {precision:.4f}")
            print(f"  - Sensitivity: {recall:.4f}")
            print(f"  - Specificity: {specificity:.4f}")
            print(f"  - F1-Score: {f1:.4f}")
            print(f"  - MCC: {mcc:.4f}")
            print(f"  - ROC AUC: {roc_auc:.4f}")
            
            return results
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e)
            }

def train_base_models_for_dataset(X_train, X_test, y_train, y_test, dataset_name, num_classes=2):
    """
    Train all three base models (ANN, LSTM, CNN) for a dataset
    
    Args:
        X_train: Training features
        X_test: Testing features  
        y_train: Training targets
        y_test: Testing targets
        dataset_name (str): Name of the dataset
        num_classes (int): Number of classes
    
    Returns:
        tuple: (base_results, base_predictions)
    """
    print(f"\nTraining base models for {dataset_name} dataset...")
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize model builder
    input_dim = X_train.shape[1]
    builder = BaseModelBuilder(input_dim=input_dim, num_classes=num_classes)
    
    # Models to train
    models_config = {
        'ANN': builder.build_ann_model,
        'LSTM': builder.build_lstm_model,
        'CNN': builder.build_cnn_model
    }
    
    base_results = {}
    base_predictions = {}
    
    # Train each model
    for model_name, model_builder in models_config.items():
        try:
            print(f"\n--- Training {model_name} ---")
            
            # Build model
            model = model_builder()
            
            # Train model
            result = builder.train_model(
                model, model_name, X_train, X_test, y_train, y_test
            )
            
            base_results[model_name] = result
            
            if result['status'] == 'success':
                base_predictions[model_name] = {
                    'train_predictions': result['predictions']['train_predictions'],
                    'test_predictions': result['predictions']['test_predictions'],
                    'train_probabilities': result['predictions']['train_probabilities'],
                    'test_probabilities': result['predictions']['test_probabilities']
                }
        
        except Exception as e:
            print(f"✗ Failed to train {model_name}: {str(e)}")
            base_results[model_name] = {
                'status': 'failed', 
                'error': str(e)
            }
    
    print(f"\nBase models training completed for {dataset_name}")
    successful_models = [name for name, result in base_results.items() 
                        if result.get('status') == 'success']
    print(f"Successfully trained: {successful_models}")
    
    return base_results, base_predictions

def save_base_models(base_results, dataset_name, models_dir='models'):
    """
    Save trained base models to disk
    
    Args:
        base_results (dict): Results from base model training
        dataset_name (str): Name of the dataset
        models_dir (str): Directory to save models
    """
    print(f"\nSaving base models for {dataset_name}...")
    
    os.makedirs(models_dir, exist_ok=True)
    dataset_dir = os.path.join(models_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    for model_name, result in base_results.items():
        if result.get('status') == 'success' and 'model' in result:
            model_path = os.path.join(dataset_dir, f'{model_name}_base_model.h5')
            try:
                result['model'].save(model_path)
                print(f"✓ Saved {model_name} model to {model_path}")
            except Exception as e:
                print(f"✗ Failed to save {model_name} model: {str(e)}")

if __name__ == "__main__":
    # Test base models with dummy data
    print("Testing base models with dummy data...")

    # Generate dummy data
    X_train = np.random.randn(100, 6)
    X_test = np.random.randn(20, 6)
    y_train = np.random.randint(0, 2, 100)
    y_test = np.random.randint(0, 2, 20)

    # Train models
    results, predictions = train_base_models_for_dataset(
        X_train, X_test, y_train, y_test,
        'TEST', num_classes=2
    )

    print("\n===== MODEL METRICS SUMMARY =====")
    for model_name, result in results.items():
        if result.get('status') == 'success':
            evals = result['evaluation']
            print(f"\nModel: {model_name}")
            print(f"  Accuracy:    {evals['accuracy']:.4f}")
            print(f"  Precision:   {evals['precision']:.4f}")
            print(f"  Recall:      {evals['sensitivity']:.4f}")
            print(f"  F1-Score:    {evals['f1_score']:.4f}")
            print(f"  MCC:         {evals['mcc']:.4f}")
            print(f"  Specificity: {evals['specificity']:.4f}")
            print(f"  ROC AUC:     {evals['roc_auc']:.4f}")
        else:
            print(f"\nModel: {model_name} - Training Failed: {result.get('error')}")
    print("\nTest completed successfully!")