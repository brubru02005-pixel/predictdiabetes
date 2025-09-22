"""
Base-Level Deep Learning Models for Ensemble Deep Learning (EDL) Diabetes Prediction System

This module implements the three base-level deep learning models as described in the paper:
"An Innovative Ensemble Deep Learning Clinical Decision Support System for Diabetes Prediction"

Models implemented:
1. Artificial Neural Network (ANN)
2. Long Short-Term Memory (LSTM) Network  
3. Convolutional Neural Network (CNN)

Each model predicts diabetes status independently and serves as input to meta-level stacking models.
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
        
        # Model hyperparameters from paper
        self.epochs = 20
        self.batch_size = 32
        self.learning_rate = 0.01
        self.validation_split = 0.2
        
        # Model storage
        self.models = {}
        self.histories = {}
    
    def build_ann_model(self):
        """
        Build Artificial Neural Network (ANN) model
        
        Returns:
            keras.Model: Compiled ANN model
        """
        print("Building ANN (Artificial Neural Network) model...")
        
        if self.num_classes == 2:
            model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(1, activation='sigmoid')
            ])
            optimizer = Adam(learning_rate=self.learning_rate)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
                layers.Dropout(0.2),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            optimizer = Adam(learning_rate=self.learning_rate)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f"ANN model built successfully:")
        print(f"- Input dimension: {self.input_dim}")
        print(f"- Output classes: {self.num_classes}")
        print(f"- Architecture: 64 -> 32 -> 16 -> {self.num_classes} neurons")
        
        return model
    
    def build_lstm_model(self):
        """
        Build Long Short-Term Memory (LSTM) model
        
        Returns:
            keras.Model: Compiled LSTM model
        """
        print("Building LSTM (Long Short-Term Memory) model...")
        
        if self.num_classes == 2:
            model = models.Sequential([
                layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
                layers.LSTM(64, return_sequences=True, dropout=0.2),
                layers.LSTM(32, dropout=0.2),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(1, activation='sigmoid')
            ])
            optimizer = Adam(learning_rate=self.learning_rate)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model = models.Sequential([
                layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
                layers.LSTM(64, return_sequences=True, dropout=0.2),
                layers.LSTM(32, dropout=0.2),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            optimizer = Adam(learning_rate=self.learning_rate)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f"LSTM model built successfully:")
        print(f"- Input dimension: {self.input_dim}")
        print(f"- Output classes: {self.num_classes}")
        print(f"- Architecture: LSTM(64) -> LSTM(32) -> Dense(16) -> {self.num_classes}")
        
        return model
    
    def build_cnn_model(self):
        """
        Build Convolutional Neural Network (CNN) model
        
        Returns:
            keras.Model: Compiled CNN model
        """
        print("Building CNN (Convolutional Neural Network) model...")
        
        # For 1D CNN on tabular data
        if self.num_classes == 2:
            model = models.Sequential([
                layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
                layers.Conv1D(32, 3, activation='relu', padding='same'),
                layers.MaxPooling1D(2),
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.MaxPooling1D(2),
                layers.Flatten(),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(1, activation='sigmoid')
            ])
            optimizer = Adam(learning_rate=self.learning_rate)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model = models.Sequential([
                layers.Reshape((self.input_dim, 1), input_shape=(self.input_dim,)),
                layers.Conv1D(32, 3, activation='relu', padding='same'),
                layers.MaxPooling1D(2),
                layers.Conv1D(64, 3, activation='relu', padding='same'),
                layers.MaxPooling1D(2),
                layers.Flatten(),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(16, activation='relu'),
                layers.Dropout(0.1),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            optimizer = Adam(learning_rate=self.learning_rate)
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        print(f"CNN model built successfully:")
        print(f"- Input dimension: {self.input_dim}")
        print(f"- Output classes: {self.num_classes}")
        print(f"- Architecture: Conv1D(32) -> Conv1D(64) -> Dense(32) -> Dense(16) -> {self.num_classes}")
        
        return model
    
    def train_model(self, model, X_train, y_train, model_name, verbose=1):
        """
        Train a base model
        
        Args:
            model: Keras model to train
            X_train: Training features
            y_train: Training targets
            model_name (str): Name of the model
            verbose (int): Verbosity level
        
        Returns:
            tuple: (trained_model, training_history)
        """
        print(f"\nTraining {model_name} model...")
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
        

        # Fix for binary classification: ensure y_train shape matches model output
        from keras.utils import to_categorical
        y_train_fit = y_train
        if self.num_classes == 2:
            # If model output is shape (None, 2) and loss is categorical, use one-hot
            if model.output_shape[-1] == 2 and model.loss == 'categorical_crossentropy':
                if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
                    y_train_fit = to_categorical(y_train, num_classes=2)
            # If model output is shape (None, 1) and loss is binary, use original labels
            elif model.output_shape[-1] == 1 and model.loss == 'binary_crossentropy':
                y_train_fit = y_train

        history = model.fit(
            X_train, y_train_fit,
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
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluate a trained model
        
        Args:
            model: Trained Keras model
            X_test: Test features
            y_test: Test targets
            model_name (str): Name of the model
        
        Returns:
            dict: Evaluation results
        """
        print(f"\nEvaluating {model_name} model...")
        
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
    
    def save_model(self, model, model_name, dataset_name):
        """
        Save trained model
        
        Args:
            model: Trained Keras model
            model_name (str): Name of the model
            dataset_name (str): Dataset identifier
        """
        model_path = f'models/{model_name}_{dataset_name}.h5'
        model.save(model_path)
        print(f"{model_name} model saved to: {model_path}")
    
    def build_and_train_all_models(self, X_train, y_train, X_test, y_test, dataset_name):
        """
        Build and train all three base models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            dataset_name (str): Dataset identifier
        
        Returns:
            dict: Results for all models
        """
        print(f"\n{'='*60}")
        print(f"TRAINING BASE-LEVEL MODELS FOR {dataset_name} DATASET")
        print(f"{'='*60}")
        
        results = {}
        model_builders = {
            'ANN': self.build_ann_model,
            'LSTM': self.build_lstm_model,
            'CNN': self.build_cnn_model
        }
        
        for model_name, builder_func in model_builders.items():
            try:
                print(f"\n{'-'*40}")
                print(f"Processing {model_name} Model")
                print(f"{'-'*40}")
                
                # Build model
                model = builder_func()
                
                # Train model
                trained_model, history = self.train_model(model, X_train, y_train, model_name)
                
                # Evaluate model
                evaluation = self.evaluate_model(trained_model, X_test, y_test, model_name)
                
                # Save model
                self.save_model(trained_model, model_name, dataset_name)
                
                # Store results
                results[model_name] = {
                    'model': trained_model,
                    'history': history,
                    'evaluation': evaluation
                }
                
                print(f"✅ {model_name} model completed successfully!")
                
            except Exception as e:
                print(f"❌ Error training {model_name} model: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results

def train_base_models_for_dataset(X_train, X_test, y_train, y_test, dataset_name, num_classes=2):
    """
    Train all base models for a specific dataset
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        dataset_name (str): Dataset identifier
        num_classes (int): Number of classes
    
    Returns:
        tuple: (results, base_predictions)
    """
    print(f"\nTraining base-level models for {dataset_name} dataset...")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Number of classes: {num_classes}")
    
    # Initialize model builder
    builder = BaseModelBuilder(
        input_dim=X_train.shape[1], 
        num_classes=num_classes,
        random_state=42
    )
    
    # Train all models
    results = builder.build_and_train_all_models(X_train, y_train, X_test, y_test, dataset_name)
    
    # Collect base model predictions for meta-level training
    base_predictions = {}
    
    for model_name, model_results in results.items():
        if 'error' not in model_results:
            # Get training predictions for meta-model training
            model = model_results['model']
            
            if num_classes > 2:
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
            else:
                train_pred = model.predict(X_train).flatten()
                test_pred = model.predict(X_test).flatten()
            
            base_predictions[model_name] = {
                'train_predictions': train_pred,
                'test_predictions': test_pred
            }
    
    # Summary
    print(f"\n{'='*50}")
    print(f"BASE MODELS TRAINING SUMMARY FOR {dataset_name}")
    print(f"{'='*50}")
    
    successful_models = 0
    for model_name, model_results in results.items():
        if 'error' not in model_results:
            accuracy = model_results['evaluation']['accuracy']
            print(f"{model_name:4s}: ✅ Accuracy = {accuracy:.4f}")
            successful_models += 1
        else:
            print(f"{model_name:4s}: ❌ Failed - {model_results['error']}")
    
    print(f"\nSuccessfully trained: {successful_models}/3 base models")
    
    # === ENSEMBLE STACKING ===
    print(f"\n{'='*50}")
    print(f"ENSEMBLE STACKING FOR {dataset_name}")
    print(f"{'='*50}")
    try:
        from stacking_models import train_stacking_models_for_dataset
        stacking_results, final_predictions = train_stacking_models_for_dataset(
            {k: v['train_predictions'] for k, v in base_predictions.items()},
            {k: v['test_predictions'] for k, v in base_predictions.items()},
            y_train, y_test, dataset_name, num_classes=num_classes
        )
        print("\nStacking Results:")
        for model_name, model_result in stacking_results.items():
            if 'error' not in model_result:
                acc = model_result['evaluation']['accuracy']
                print(f"{model_name:10s}: ✅ Stacking Accuracy = {acc:.4f}")
            else:
                print(f"{model_name:10s}: ❌ Failed - {model_result['error']}")
        # Print best stacking model summary
        best_model = None
        best_acc = 0
        for model_name, model_result in stacking_results.items():
            if 'error' not in model_result and model_result['evaluation']['accuracy'] > best_acc:
                best_acc = model_result['evaluation']['accuracy']
                best_model = model_name
        if best_model:
            print(f"\nBest stacking model: {best_model} with accuracy: {best_acc:.4f}")
    except Exception as e:
        print(f"[Stacking Error] {str(e)}")
    return results, base_predictions

def main():
    """
    Test base models training
    """
    from data_preprocessing import DiabetesDataPreprocessor
    from feature_selection import select_features_for_dataset
    
    print("Testing Base-Level Models Training")
    print("=" * 50)
    
    # Load and preprocess data
    preprocessor = DiabetesDataPreprocessor()
    
    # Test with ALL datasets combined
    try:
        # Load and combine all datasets
        data_files = [
            os.path.join('data', 'Dataset of Diabetes .csv'),
            os.path.join('data', 'diabetes.csv'),
            os.path.join('data', 'pima.csv.csv')
        ]
        import pandas as pd
        df_list = [pd.read_csv(f) for f in data_files]
        combined_df = pd.concat(df_list, ignore_index=True)
        # Save combined for reproducibility (optional)
        combined_path = os.path.join('data', 'combined_diabetes.csv')
        combined_df.to_csv(combined_path, index=False)
        result = preprocessor.preprocess_dataset("COMBINED", apply_smote_flag=True)
        if result is not None:
            X_train, X_test, y_train, y_test, info = result
            # Use mutual information feature selection (top 6 features)
            print("Selecting top 6 features using mutual information...")
            X_train_selected, X_test_selected, selector, summary = select_features_for_dataset(
                X_train, X_test, y_train, 'COMBINED', feature_names=info['features'], method='mutual_info', k=6
            )
            print(f"Selected features: {[f['feature'] for f in summary['top_3_features']]}")
            # Train base models
            results, base_predictions = train_base_models_for_dataset(
                X_train_selected, X_test_selected, y_train, y_test, 
                'COMBINED', num_classes=len(info['classes'])
            )
            print("\n✅ Base models training test completed successfully!")
            # Show base predictions shape for meta-models
            print("\nBase predictions for meta-level training:")
            for model_name, preds in base_predictions.items():
                train_shape = preds['train_predictions'].shape
                test_shape = preds['test_predictions'].shape
                print(f"  {model_name}: Train {train_shape}, Test {test_shape}")
        else:
            print("❌ Failed to preprocess dataset")
    except Exception as e:
        print(f"❌ Error in base models training test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
