"""
Enhanced API Endpoints for Diabetes Prediction System

Provides RESTful API endpoints for:
1. PIMA diabetes predictions with enhanced clinical assessment
2. DDFH diabetes predictions with multi-class classification
3. Unified prediction interface with automatic model selection
4. Model status and health checks
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import traceback
from unified_predictor import UnifiedDiabetesPredictor

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Initialize unified predictor
try:
    predictor = UnifiedDiabetesPredictor()
    print("✓ API: Enhanced Unified Predictor loaded successfully")
except Exception as e:
    print(f"⚠️ API Warning: Could not load unified predictor: {e}")
    predictor = None

@api_bp.route('/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'predictor_loaded': predictor is not None,
        'version': '2.0-enhanced'
    })

@api_bp.route('/models/status', methods=['GET'])
def model_status():
    """Get status of loaded models"""
    if not predictor:
        return jsonify({'error': 'Predictor not loaded'}), 503
    
    status = {
        'timestamp': datetime.now().isoformat(),
        'predictor_status': 'loaded',
        'datasets': {}
    }
    
    for dataset, models in predictor.models.items():
        status['datasets'][dataset] = {
            'loaded_models': list(models.keys()),
            'total_models': len(models),
            'status': 'ready' if models else 'no_models'
        }
    
    return jsonify(status)

@api_bp.route('/predict', methods=['POST'])
def predict():
    """
    Unified prediction endpoint
    Automatically selects best model based on available features
    """
    try:
        if not predictor:
            return jsonify({
                'error': 'Prediction service unavailable',
                'message': 'Unified predictor not loaded'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body must contain JSON data'
            }), 400
        
        # Use unified predictor with automatic model selection
        result = predictor.predict(data)
        
        # Add API metadata
        result['api_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': 'unified_predict',
            'version': '2.0-enhanced'
        }
        
        return jsonify(result)
        
    except Exception as e:
        error_details = {
            'error': 'Prediction failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'traceback': traceback.format_exc() if request.args.get('debug') else None
        }
        return jsonify(error_details), 500

@api_bp.route('/predict/pima', methods=['POST'])
def predict_pima():
    """
    PIMA-specific prediction endpoint
    Enhanced with clinical risk assessment
    """
    try:
        if not predictor:
            return jsonify({
                'error': 'Prediction service unavailable'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Validate required PIMA fields
        required_fields = ['glucose', 'bmi', 'age']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400
        
        # Force PIMA prediction by ensuring glucose is present and hba1c is not
        pima_data = data.copy()
        if 'hba1c' in pima_data:
            del pima_data['hba1c']  # Remove HbA1c to force PIMA prediction
        
        result = predictor.predict(pima_data)
        
        # Add PIMA-specific metadata
        result['api_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': 'pima_predict',
            'dataset': 'PIMA',
            'version': '2.0-enhanced'
        }
        
        return jsonify(result)
        
    except Exception as e:
        error_details = {
            'error': 'PIMA prediction failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_details), 500

@api_bp.route('/predict/ddfh', methods=['POST'])
def predict_ddfh():
    """
    DDFH-specific prediction endpoint
    Multi-class diabetes classification
    """
    try:
        if not predictor:
            return jsonify({
                'error': 'Prediction service unavailable'
            }), 503
        
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Validate required DDFH fields
        required_fields = ['hba1c', 'bmi', 'age']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing_fields': missing_fields,
                'required_fields': required_fields
            }), 400
        
        result = predictor.predict(data)
        
        # Add DDFH-specific metadata
        result['api_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': 'ddfh_predict',
            'dataset': 'DDFH', 
            'version': '2.0-enhanced'
        }
        
        return jsonify(result)
        
    except Exception as e:
        error_details = {
            'error': 'DDFH prediction failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_details), 500

@api_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    Process multiple predictions in one request
    """
    try:
        if not predictor:
            return jsonify({
                'error': 'Prediction service unavailable'
            }), 503
        
        data = request.get_json()
        if not data or 'predictions' not in data:
            return jsonify({
                'error': 'Invalid batch request',
                'message': 'Request must contain "predictions" array'
            }), 400
        
        predictions_data = data['predictions']
        if not isinstance(predictions_data, list):
            return jsonify({
                'error': 'Invalid predictions format',
                'message': 'Predictions must be an array'
            }), 400
        
        # Limit batch size for performance
        max_batch_size = 100
        if len(predictions_data) > max_batch_size:
            return jsonify({
                'error': 'Batch too large',
                'message': f'Maximum batch size is {max_batch_size}',
                'received': len(predictions_data)
            }), 400
        
        results = []
        errors = []
        
        for i, pred_data in enumerate(predictions_data):
            try:
                result = predictor.predict(pred_data)
                result['batch_index'] = i
                results.append(result)
            except Exception as e:
                errors.append({
                    'batch_index': i,
                    'error': str(e),
                    'input_data': pred_data
                })
        
        response = {
            'batch_results': {
                'timestamp': datetime.now().isoformat(),
                'total_requests': len(predictions_data),
                'successful_predictions': len(results),
                'failed_predictions': len(errors),
                'results': results
            }
        }
        
        if errors:
            response['batch_results']['errors'] = errors
        
        return jsonify(response)
        
    except Exception as e:
        error_details = {
            'error': 'Batch prediction failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_details), 500

@api_bp.route('/features/info', methods=['GET'])
def feature_info():
    """
    Get information about expected features for each dataset
    """
    return jsonify({
        'feature_info': {
            'PIMA': {
                'required_features': ['glucose', 'bmi', 'age'],
                'optional_features': ['pregnancies', 'diabetes_pedigree', 'skin_thickness', 'blood_pressure', 'insulin'],
                'feature_descriptions': {
                    'glucose': 'Blood glucose level (mg/dL)',
                    'bmi': 'Body Mass Index (kg/m²)',
                    'age': 'Age in years',
                    'pregnancies': 'Number of pregnancies',
                    'diabetes_pedigree': 'Diabetes pedigree function (0-3)',
                    'skin_thickness': 'Triceps skin fold thickness (mm)',
                    'blood_pressure': 'Diastolic blood pressure (mmHg)',
                    'insulin': '2-hour serum insulin (μU/ml)'
                }
            },
            'DDFH': {
                'required_features': ['hba1c', 'bmi', 'age'],
                'optional_features': ['glucose', 'tg', 'gender'],
                'feature_descriptions': {
                    'hba1c': 'Hemoglobin A1C percentage (%)',
                    'bmi': 'Body Mass Index (kg/m²)',
                    'age': 'Age in years',
                    'glucose': 'Blood glucose level (mg/dL)',
                    'tg': 'Triglycerides level (mg/dL)',
                    'gender': 'Gender (M/F)'
                }
            }
        },
        'api_info': {
            'version': '2.0-enhanced',
            'features': [
                'Automatic model selection',
                'Enhanced clinical risk assessment',
                'Evidence-based feature weighting',
                'ADA-compliant medical validation',
                'Detailed clinical interpretations'
            ]
        }
    })

@api_bp.errorhandler(404)
def api_not_found(error):
    """API 404 error handler"""
    return jsonify({
        'error': 'API endpoint not found',
        'message': 'The requested API endpoint does not exist',
        'timestamp': datetime.now().isoformat()
    }), 404

@api_bp.errorhandler(500)
def api_internal_error(error):
    """API 500 error handler"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred in the API',
        'timestamp': datetime.now().isoformat()
    }), 500