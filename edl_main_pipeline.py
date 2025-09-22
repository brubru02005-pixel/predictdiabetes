"""
Main Pipeline for Ensemble Deep Learning (EDL) Diabetes Prediction System

This module implements the complete EDL pipeline as described in the paper:
"An Innovative Ensemble Deep Learning Clinical Decision Support System for Diabetes Prediction"

Pipeline steps:
1. Data acquisition and preprocessing
2. Feature selection using Extra Tree Classifier
3. Base-level model training (ANN, LSTM, CNN)
4. Meta-level stacking model training (Stack-ANN, Stack-LSTM, Stack-CNN)
5. Comprehensive evaluation and comparison
6. Results reporting and model saving
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all components
from data_preprocessing import DiabetesDataPreprocessor
from feature_selection import select_features_for_dataset
from base_models import train_base_models_for_dataset
from stacking_models import train_stacking_models_for_dataset

class EDLPipeline:
    """
    Complete Ensemble Deep Learning Pipeline
    """
    
    def __init__(self, config=None):
        """
        Initialize EDL Pipeline
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or self.get_default_config()
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def get_default_config(self):
        """
        Get default configuration for the pipeline
        
        Returns:
            dict: Default configuration
        """
        return {
            'datasets': ['PIMA', 'DDFH', 'IDPD'],
            'apply_smote': True,
            'feature_selection': {
                'method': 'top_k',
                'k': 6
            },
            'models': {
                'base_models': ['ANN', 'LSTM', 'CNN'],
                'stacking_models': ['Stack-ANN', 'Stack-LSTM', 'Stack-CNN']
            },
            'evaluation': {
                'metrics': ['accuracy', 'precision', 'sensitivity', 'specificity', 
                          'f1_score', 'mcc', 'roc_auc']
            },
            'save_models': True,
            'save_results': True,
            'results_dir': 'results',
            'models_dir': 'models'
        }
    
    def setup_directories(self):
        """
        Setup necessary directories
        """
        directories = [
            self.config['results_dir'],
            self.config['models_dir']
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Directory created/verified: {directory}")
    
    def run_single_dataset(self, dataset_name):
        """
        Run complete pipeline for a single dataset
        
        Args:
            dataset_name (str): Name of dataset to process
        
        Returns:
            dict: Results for the dataset
        """
        print(f"\n{'#'*80}")
        print(f"PROCESSING {dataset_name} DATASET")
        print(f"{'#'*80}")
        
        dataset_start_time = time.time()
        dataset_results = {
            'dataset': dataset_name,
            'start_time': dataset_start_time,
            'status': 'processing'
        }
        
        try:
            # Step 1: Data preprocessing
            print(f"\n1. DATA PREPROCESSING - {dataset_name}")
            print("-" * 50)
            
            preprocessor = DiabetesDataPreprocessor()
            preprocessing_result = preprocessor.preprocess_dataset(
                dataset_name, 
                apply_smote_flag=self.config['apply_smote']
            )
            
            if preprocessing_result is None:
                raise Exception(f"Failed to preprocess {dataset_name} dataset")
            
            X_train, X_test, y_train, y_test, dataset_info = preprocessing_result
            dataset_results['preprocessing'] = {
                'original_shape': dataset_info['original_shape'],
                'processed_shape': dataset_info['processed_shape'],
                'train_size': dataset_info['train_size'],
                'test_size': dataset_info['test_size'],
                'features': dataset_info['features'],
                'classes': dataset_info['classes']
            }
            
            print(f"✓ {dataset_name} preprocessing completed")
            
            # Step 2: Feature selection
            print(f"\n2. FEATURE SELECTION - {dataset_name}")
            print("-" * 50)
            
            X_train_selected, X_test_selected, selector, selection_summary = select_features_for_dataset(
                X_train, X_test, y_train, dataset_name,
                feature_names=dataset_info['features'],
                method=self.config['feature_selection']['method'],
                k=self.config['feature_selection']['k']
            )
            
            dataset_results['feature_selection'] = {
                'original_features': selection_summary['total_features'],
                'selected_features': selection_summary['selected_features'],
                'top_features': selection_summary['top_3_features']
            }
            
            print(f"✓ {dataset_name} feature selection completed")
            
            # Step 3: Base models training
            print(f"\n3. BASE MODELS TRAINING - {dataset_name}")
            print("-" * 50)
            
            base_results, base_predictions = train_base_models_for_dataset(
                X_train_selected, X_test_selected, y_train, y_test,
                dataset_name, num_classes=len(dataset_info['classes'])
            )
            
            dataset_results['base_models'] = {}
            for model_name, model_result in base_results.items():
                if 'error' not in model_result:
                    accuracy = model_result['evaluation']['accuracy']
                    dataset_results['base_models'][model_name] = {
                        'accuracy': accuracy,
                        'status': 'success'
                    }
                    print(f"✓ {model_name} trained successfully: {accuracy:.4f}")
                else:
                    dataset_results['base_models'][model_name] = {
                        'status': 'failed',
                        'error': model_result['error']
                    }
                    print(f"✗ {model_name} failed: {model_result['error']}")
            
            # Step 4: Stacking models training
            if len(base_predictions) > 0:
                print(f"\n4. STACKING MODELS TRAINING - {dataset_name}")
                print("-" * 50)
                
                # Prepare stacking data
                base_train_preds = {name: preds['train_predictions'] 
                                  for name, preds in base_predictions.items()}
                base_test_preds = {name: preds['test_predictions'] 
                                 for name, preds in base_predictions.items()}
                
                stacking_results, final_predictions = train_stacking_models_for_dataset(
                    base_train_preds, base_test_preds, y_train, y_test,
                    dataset_name, num_classes=len(dataset_info['classes'])
                )
                
                dataset_results['stacking_models'] = {}
                for model_name, model_result in stacking_results.items():
                    if 'error' not in model_result:
                        accuracy = model_result['evaluation']['accuracy']
                        dataset_results['stacking_models'][model_name] = {
                            'accuracy': accuracy,
                            'status': 'success'
                        }
                        print(f"✓ {model_name} trained successfully: {accuracy:.4f}")
                    else:
                        dataset_results['stacking_models'][model_name] = {
                            'status': 'failed',
                            'error': model_result['error']
                        }
                        print(f"✗ {model_name} failed: {model_result['error']}")
                
            else:
                print(f"\n✗ No base model predictions available for stacking - {dataset_name}")
                dataset_results['stacking_models'] = {'status': 'skipped', 'reason': 'no_base_predictions'}
            
            # Step 5: Performance summary
            print(f"\n5. PERFORMANCE SUMMARY - {dataset_name}")
            print("-" * 50)
            
            # Find best models
            best_base_accuracy = 0
            best_base_model = None
            for model_name, model_info in dataset_results['base_models'].items():
                if model_info['status'] == 'success' and model_info['accuracy'] > best_base_accuracy:
                    best_base_accuracy = model_info['accuracy']
                    best_base_model = model_name
            

            best_stacking_accuracy = 0
            best_stacking_model = None
            stacking_models = dataset_results.get('stacking_models', {})
            # If stacking_models is a dict of models (not skipped), iterate properly
            if isinstance(stacking_models, dict) and not ('status' in stacking_models and stacking_models['status'] == 'skipped'):
                for model_name, model_info in stacking_models.items():
                    if isinstance(model_info, dict) and model_info.get('status') == 'success' and model_info.get('accuracy', 0) > best_stacking_accuracy:
                        best_stacking_accuracy = model_info['accuracy']
                        best_stacking_model = model_name

            dataset_results['performance_summary'] = {
                'best_base_model': best_base_model,
                'best_base_accuracy': best_base_accuracy,
                'best_stacking_model': best_stacking_model,
                'best_stacking_accuracy': best_stacking_accuracy,
                'improvement': best_stacking_accuracy - best_base_accuracy if best_stacking_accuracy > 0 else 0
            }

            print(f"Best Base Model: {best_base_model} ({best_base_accuracy:.4f})")
            if best_stacking_model:
                print(f"Best Stacking Model: {best_stacking_model} ({best_stacking_accuracy:.4f})")
                improvement = best_stacking_accuracy - best_base_accuracy
                print(f"Improvement: {improvement:.4f} ({improvement*100:.2f}%)")

            dataset_results['status'] = 'completed'
            dataset_results['end_time'] = time.time()
            dataset_results['duration'] = dataset_results['end_time'] - dataset_start_time

            print(f"\n✅ {dataset_name} processing completed in {dataset_results['duration']:.2f} seconds")
            
        except Exception as e:
            dataset_results['status'] = 'failed'
            dataset_results['error'] = str(e)
            dataset_results['end_time'] = time.time()
            dataset_results['duration'] = dataset_results['end_time'] - dataset_start_time
            
            print(f"\n❌ {dataset_name} processing failed: {str(e)}")
        
        return dataset_results
    
    def run_complete_pipeline(self):
        """
        Run complete EDL pipeline for all configured datasets
        
        Returns:
            dict: Complete results
        """
        print("="*80)
        print("ENSEMBLE DEEP LEARNING (EDL) DIABETES PREDICTION SYSTEM")
        print("="*80)
        print("Implementation based on the paper:")
        print("'An Innovative Ensemble Deep Learning Clinical Decision Support System for Diabetes Prediction'")
        print("="*80)
        
        self.start_time = time.time()
        
        # Setup directories
        self.setup_directories()
        
        # Process each dataset
        all_results = {}
        successful_datasets = 0
        
        for dataset_name in self.config['datasets']:
            try:
                dataset_results = self.run_single_dataset(dataset_name)
                all_results[dataset_name] = dataset_results
                
                if dataset_results['status'] == 'completed':
                    successful_datasets += 1
                    
            except Exception as e:
                print(f"❌ Critical error processing {dataset_name}: {str(e)}")
                all_results[dataset_name] = {
                    'status': 'critical_error',
                    'error': str(e),
                    'dataset': dataset_name
                }
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Generate final report
        self.generate_final_report(all_results, successful_datasets, total_duration)
        
        # Save results
        if self.config['save_results']:
            self.save_pipeline_results(all_results)
        
        return all_results
    
    def generate_final_report(self, all_results, successful_datasets, total_duration):
        """
        Generate final pipeline report
        
        Args:
            all_results (dict): All dataset results
            successful_datasets (int): Number of successful datasets
            total_duration (float): Total execution time
        """
        print(f"\n{'#'*80}")
        print("FINAL PIPELINE REPORT")
        print(f"{'#'*80}")
        
        print(f"\nExecution Summary:")
        print(f"- Total datasets processed: {len(self.config['datasets'])}")
        print(f"- Successful datasets: {successful_datasets}")
        print(f"- Failed datasets: {len(self.config['datasets']) - successful_datasets}")
        print(f"- Total execution time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)")
        
        print(f"\nDataset Results:")
        print("-" * 40)
        
        for dataset_name, results in all_results.items():
            print(f"\n{dataset_name}:")
            print(f"  Status: {results['status']}")
            
            if results['status'] == 'completed':
                summary = results['performance_summary']
                print(f"  Best Base Model: {summary['best_base_model']} ({summary['best_base_accuracy']:.4f})")
                if summary['best_stacking_model']:
                    print(f"  Best Stacking Model: {summary['best_stacking_model']} ({summary['best_stacking_accuracy']:.4f})")
                    print(f"  Improvement: {summary['improvement']:.4f} ({summary['improvement']*100:.2f}%)")
                print(f"  Duration: {results['duration']:.2f}s")
            elif results['status'] == 'failed':
                print(f"  Error: {results.get('error', 'Unknown error')}")
        
        # Overall performance summary
        if successful_datasets > 0:
            print(f"\nOverall Performance Summary:")
            print("-" * 40)
            
            total_improvement = 0
            valid_improvements = 0
            
            for dataset_name, results in all_results.items():
                if results['status'] == 'completed':
                    summary = results['performance_summary']
                    if summary['improvement'] > 0:
                        total_improvement += summary['improvement']
                        valid_improvements += 1
                        
            if valid_improvements > 0:
                avg_improvement = total_improvement / valid_improvements
                print(f"Average stacking improvement: {avg_improvement:.4f} ({avg_improvement*100:.2f}%)")
        
        print(f"\n✅ Pipeline execution completed!")
    
    def save_pipeline_results(self, all_results):
        """
        Save pipeline results to files
        
        Args:
            all_results (dict): All dataset results
        """
        print(f"\nSaving pipeline results...")
        
        # Create summary DataFrame
        summary_data = []
        
        for dataset_name, results in all_results.items():
            if results['status'] == 'completed':
                summary = results['performance_summary']
                preprocessing = results['preprocessing']
                
                row = {
                    'Dataset': dataset_name,
                    'Status': results['status'],
                    'Original_Samples': preprocessing['original_shape'][0],
                    'Original_Features': preprocessing['original_shape'][1],
                    'Processed_Samples': preprocessing['processed_shape'][0],
                    'Selected_Features': preprocessing['processed_shape'][1],
                    'Best_Base_Model': summary['best_base_model'],
                    'Best_Base_Accuracy': summary['best_base_accuracy'],
                    'Best_Stacking_Model': summary['best_stacking_model'],
                    'Best_Stacking_Accuracy': summary['best_stacking_accuracy'],
                    'Improvement': summary['improvement'],
                    'Improvement_Percent': summary['improvement'] * 100,
                    'Duration_Seconds': results['duration']
                }
            else:
                row = {
                    'Dataset': dataset_name,
                    'Status': results['status'],
                    'Error': results.get('error', 'Unknown'),
                    'Duration_Seconds': results.get('duration', 0)
                }
            
            summary_data.append(row)
        
        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_path = f"{self.config['results_dir']}/pipeline_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Pipeline summary saved to: {summary_path}")
        
        # Save detailed results as JSON
        import json
        results_path = f"{self.config['results_dir']}/detailed_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"✓ Detailed results saved to: {results_path}")

def main():
    """
    Main function to run the complete EDL pipeline
    """
    # Configuration
    config = {
        'datasets': ['PIMA', 'DDFH'],  # Test with available datasets
        'apply_smote': True,
        'feature_selection': {
            'method': 'top_k',
            'k': 6
        },
        'save_models': True,
        'save_results': True,
        'results_dir': 'results',
        'models_dir': 'models'
    }
    
    print("Starting Ensemble Deep Learning (EDL) Pipeline Test...")
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize and run pipeline
    pipeline = EDLPipeline(config)
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "="*80)
    print("EDL PIPELINE TEST COMPLETED")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()