"""
Flask Web Application for Ensemble Deep Learning (EDL) Diabetes Prediction System

This web application integrates the EDL system with a user-friendly interface for:
1. PIMA dataset predictions (binary classification)
2. DDFH dataset predictions (multi-class classification)
3. Real-time predictions using trained models
4. Comprehensive results visualization
"""

import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import logging
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show ERROR messages

# Import our unified predictor and clinic system
from unified_predictor import UnifiedDiabetesPredictor
from clinic_deployment import ClinicDiabetesSystem
from api import api_bp

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Register API blueprint
app.register_blueprint(api_bp)

# Initialize clinical system (primary) and unified predictor (fallback)
try:
    clinic_system = ClinicDiabetesSystem()
    unified_predictor = UnifiedDiabetesPredictor()
    print(f"✅ Clinic Diabetes System loaded successfully")
    print(f"  🏥 Primary: DDFH Model (98.82% Clinical Grade)")
    print(f"  🔬 Secondary: PIMA Model (95% Screening)")
    print(f"  📊 Legacy: Unified Predictor Available")
except Exception as e:
    print(f"⚠️ Warning: Could not load clinic system: {e}")
    clinic_system = None
    unified_predictor = None
    
# Old prediction methods removed - now using enhanced unified_predictor


# Simple user storage (in production, use a database)
users = {}

@app.route('/')
def index():
    """Home page"""
    return render_template('base.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and check_password_hash(users[username]['password'], password):
            session['user'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('predict'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form.get('email', '')
        
        if username in users:
            flash('Username already exists', 'error')
        else:
            users[username] = {
                'password': generate_password_hash(password),
                'email': email,
                'created_at': datetime.now().isoformat()
            }
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """Logout"""
    session.pop('user', None)
    flash('Logged out successfully', 'info')
    return redirect(url_for('index'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Main prediction page - clinic-optimized workflow"""
    if request.method == 'POST':
        prediction_type = request.form.get('prediction_type', 'clinical')
        
        try:
            if prediction_type == 'clinical' or prediction_type == 'ddfh':
                # Clinical assessment using DDFH model (98.82% accuracy)
                input_data = {
                    'age': float(request.form['age']),
                    'bmi': float(request.form['bmi']),
                    'hba1c': float(request.form['hba1c']),
                    'glucose': float(request.form.get('glucose', 100)),
                    'tg': float(request.form.get('tg', 150)),
                    'gender': request.form.get('gender', 'M')
                }
                
                # Use clinic system if available, fallback to unified predictor
                if clinic_system:
                    result = clinic_system.clinical_assessment(input_data, 'clinical')
                else:
                    result = unified_predictor.predict(input_data)
                    result['model_type'] = 'Legacy DDFH'
                
            elif prediction_type == 'screening' or prediction_type == 'pima':
                # Screening using PIMA model (95% accuracy)
                input_data = {
                    'pregnancies': float(request.form['pregnancies']),
                    'glucose': float(request.form['glucose']),
                    'blood_pressure': float(request.form['blood_pressure']),
                    'skin_thickness': float(request.form['skin_thickness']),
                    'insulin': float(request.form['insulin']),
                    'bmi': float(request.form['bmi']),
                    'diabetes_pedigree': float(request.form['diabetes_pedigree']),
                    'age': float(request.form['age'])
                }
                
                # Use clinic system if available, fallback to unified predictor
                if clinic_system:
                    result = clinic_system.clinical_assessment(input_data, 'screening')
                else:
                    result = unified_predictor.predict(input_data)
                    result['model_type'] = 'Legacy PIMA'
                
            else:
                result = {'success': False, 'error': 'Invalid prediction type'}
            
            # Add metadata for display
            result['input_data'] = input_data
            result['prediction_type'] = prediction_type
            result['timestamp'] = datetime.now().isoformat()
            result['clinic_mode'] = clinic_system is not None
            
            return render_template('result_enhanced.html', result=result)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash(f'Error processing prediction: {str(e)}', 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')

@app.route('/predict_enhanced', methods=['GET', 'POST'])
def predict_enhanced():
    """Enhanced prediction route handling both PIMA and DDFH models"""
    if request.method == 'POST':
        try:
            prediction_type = request.form.get('prediction_type', 'ddfh')
            
            if prediction_type == 'ddfh':
                # DDFH model input (simplified to match available features)
                input_data = {
                    'hba1c': float(request.form.get('hba1c', 5.5)),
                    'bmi': float(request.form.get('bmi', 25.0)),
                    'age': float(request.form.get('age', 30)),
                    'tg': float(request.form.get('tg', 150)),
                    'gender': request.form.get('gender', 'M')
                }
                # Use DDFH prediction method
                result = unified_predictor.predict(input_data)
                result['prediction_type'] = 'ddfh'
                # (Optional) Add XAI for DDFH if desired
            else:  # PIMA model
                input_data = {
                    'pregnancies': float(request.form.get('pregnancies', 0)),
                    'glucose': float(request.form.get('glucose', 120)),
                    'blood_pressure': float(request.form.get('blood_pressure', 80)),
                    'skin_thickness': float(request.form.get('skin_thickness', 20)),
                    'insulin': float(request.form.get('insulin', 80)),
                    'bmi': float(request.form.get('bmi', 25.0)),
                    'diabetes_pedigree': float(request.form.get('diabetes_pedigree', 0.5)),
                    'age': float(request.form.get('age', 30))
                }
                # Use enhanced PIMA prediction method
                result = unified_predictor.predict(input_data)
                result['prediction_type'] = 'pima'
                # XAI: SHAP explanation for PIMA ANN model
                try:
                    from xai_explanations import explain_prediction
                    model = unified_predictor.models['PIMA'].get('ANN')
                    if model:
                        import numpy as np
                        X = np.array([[
                            input_data['pregnancies'],
                            input_data['glucose'],
                            input_data['blood_pressure'],
                            input_data['skin_thickness'],
                            input_data['insulin'],
                            input_data['bmi'],
                            input_data['diabetes_pedigree'],
                            input_data['age']
                        ]])
                        shap_values, expected_value = explain_prediction(model, X)
                        result['shap_values'] = shap_values[0][0].tolist() if isinstance(shap_values, list) else shap_values[0].tolist()
                        result['expected_value'] = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value
                        result['feature_names'] = ['Pregnancies','Glucose','Blood Pressure','Skin Thickness','Insulin','BMI','Diabetes Pedigree','Age']
                except Exception as ex:
                    logger.error(f"XAI explanation failed: {ex}")
            result['input_data'] = input_data
            logger.info(f"Enhanced {prediction_type.upper()} prediction completed: {result.get('class_name', 'Unknown')}")
            return render_template('result_enhanced.html', result=result)
            
        except Exception as e:
            logger.error(f"Enhanced prediction error: {str(e)}")
            return render_template('predict_enhanced.html', error=f"Prediction failed: {str(e)}")
    
    return render_template('predict_enhanced.html')


@app.route('/dashboard')
def dashboard():
    """Interactive dashboard for analytics and model management"""
    return render_template('dashboard.html')

@app.route('/clinical', methods=['GET', 'POST'])
def clinical_assessment():
    """Clinical-grade diabetes assessment using clinic deployment system"""
    if request.method == 'POST':
        try:
            # Extract clinical input data
            input_data = {
                'hba1c': float(request.form.get('hba1c', 0)),
                'bmi': float(request.form.get('bmi', 0)),
                'age': float(request.form.get('age', 0)),
                'tg': float(request.form.get('tg', 150)),
                'gender': request.form.get('gender', 'M')
            }
            
            # Perform clinic-grade assessment
            if clinic_system:
                result = clinic_system.clinical_assessment(input_data, 'clinical')
                
                # Generate comprehensive clinical report
                if result.get('success'):
                    patient_info = {
                        'patient_id': f"P{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        'assessment_date': datetime.now().isoformat()
                    }
                    clinical_report = clinic_system.generate_clinical_report(patient_info, result)
                    result['clinical_report'] = clinical_report
                    
                logger.info(f"Clinic assessment completed: {result.get('ai_prediction', 'Unknown')}")
                return render_template('clinical_results.html', result=result)
            else:
                # Fallback to legacy clinical predictor
                from clinical_deployment import ClinicalDiabetesPredictor
                clinical_predictor = ClinicalDiabetesPredictor()
                result = clinical_predictor.clinical_prediction(input_data)
                return render_template('clinical_results.html', result=result)
                
        except Exception as e:
            logger.error(f"Clinical assessment error: {str(e)}")
            return render_template('clinical_assessment.html', error=f"Assessment failed: {str(e)}")
    
    return render_template('clinical_assessment.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for clinic-optimized predictions"""
    try:
        data = request.get_json()
        assessment_type = data.get('type', 'clinical')
        
        # Determine assessment type based on data format if not specified
        if assessment_type == 'auto':
            assessment_type = 'clinical' if 'hba1c' in data else 'screening'
        
        # Use clinic system if available
        if clinic_system:
            if assessment_type in ['clinical', 'ddfh']:
                result = clinic_system.clinical_assessment(data, 'clinical')
            elif assessment_type in ['screening', 'pima']:
                result = clinic_system.clinical_assessment(data, 'screening')
            else:
                result = {'success': False, 'error': f'Invalid assessment type: {assessment_type}'}
        else:
            # Fallback to unified predictor
            result = unified_predictor.predict(data)
            result['model_type'] = 'Legacy Unified'
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/models/status')
def models_status():
    """Check status of loaded models in clinic deployment"""
    status = {
        'clinic_system': {
            'available': clinic_system is not None,
            'primary_model': 'DDFH (98.82% Clinical Grade)' if clinic_system else 'Not Available',
            'secondary_model': 'PIMA (95% Screening)' if clinic_system else 'Not Available',
            'deployment_mode': 'Clinical Deployment' if clinic_system else 'Legacy Mode'
        }
    }
    
    if unified_predictor:
        for dataset, models in unified_predictor.models.items():
            status[f'legacy_{dataset}'] = {
                'loaded_models': list(models.keys()),
                'total_models': len(models),
                'status': 'Available as fallback'
            }
    
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return render_template('base.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('base.html'), 500

if __name__ == '__main__':
    print("Starting Clinical Diabetes Prediction System...")
    print("="*70)
    print("🏥 CLINICAL DIABETES PREDICTION SYSTEM FOR HEALTHCARE")
    print("="*70)
    
    if clinic_system:
        print("🎯 CLINIC DEPLOYMENT MODE - CLINICAL GRADE MODELS")
        print(f"  🩺 Primary: DDFH Model (98.82% Accuracy) - Clinical Diagnosis")
        print(f"  🔬 Secondary: PIMA Model (95% Accuracy) - Population Screening")
        print(f"  📊 Workflow: Automated Clinical Assessment")
        print(f"  📈 Features: ADA-Compliant, Billing Codes, Patient Instructions")
        print(f"  🚫 Combined Model: Disabled (Clinical Use Only)")
    elif unified_predictor:
        print("⚠️ LEGACY MODE - FALLBACK TO UNIFIED PREDICTOR")
        pima_models = len(unified_predictor.models.get('PIMA', {}))
        ddfh_models = len(unified_predictor.models.get('DDFH', {}))
        print(f"  📊 Legacy Models: PIMA({pima_models}) + DDFH({ddfh_models})")
    else:
        print("❌ ERROR: No prediction models available")
    
    print("\n🌐 ACCESS POINTS:")
    print(f"  Web Interface: http://localhost:5000")
    print(f"  Clinical API: http://localhost:5000/api/predict")
    print(f"  Model Status: http://localhost:5000/models/status")
    print("="*70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)