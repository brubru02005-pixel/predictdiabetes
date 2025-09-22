"""
Flask Web Application for Unified Diabetes Prediction System

This web application provides accurate diabetes risk assessment using:
1. Trained neural network models (PIMA & DDFH datasets)
2. Clinical validation based on ADA guidelines
3. Unified prediction system with medical oversight
"""

import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our unified predictor
from unified_predictor import UnifiedDiabetesPredictor

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Initialize unified predictor
print("Initializing Unified Diabetes Prediction System...")
unified_predictor = UnifiedDiabetesPredictor()

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
    """Main prediction page (Standard PIMA model)"""
    if request.method == 'POST':
        try:
            # Extract PIMA features
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
            
            # Get prediction using unified predictor
            result = unified_predictor.predict(input_data)
            
            # Add metadata for display
            result['input_data'] = input_data
            result['prediction_type'] = 'pima'
            result['timestamp'] = datetime.now().isoformat()
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            flash(f'Error processing prediction: {str(e)}', 'error')
            return render_template('predict.html')
    
    return render_template('predict.html')

@app.route('/predict/enhanced', methods=['GET', 'POST'])
def predict_enhanced():
    """Enhanced prediction page with both DDFH and PIMA models"""
    if request.method == 'POST':
        try:
            prediction_type = request.form.get('prediction_type', 'ddfh')
            
            if prediction_type == 'ddfh':
                # DDFH dataset prediction (enhanced clinical model)
                input_data = {
                    'age': float(request.form['age']),
                    'bmi': float(request.form['bmi']),
                    'hba1c': float(request.form['hba1c']),
                    'glucose': float(request.form.get('glucose', 100)),
                    'tg': float(request.form.get('tg', 150)),
                    'gender': request.form.get('gender', 'M')
                }
                
            elif prediction_type == 'pima':
                # PIMA dataset prediction (standard model)
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
            else:
                input_data = {}
            
            # Get prediction using unified predictor
            result = unified_predictor.predict(input_data)
            
            # Add metadata for display
            result['input_data'] = input_data
            result['prediction_type'] = prediction_type
            result['timestamp'] = datetime.now().isoformat()
            
            return render_template('result.html', result=result)
            
        except Exception as e:
            print(f"Error in enhanced prediction: {e}")
            flash(f'Error processing prediction: {str(e)}', 'error')
            return render_template('predict_enhanced.html')
    
    return render_template('predict_enhanced.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        # Use unified predictor for all predictions
        result = unified_predictor.predict(data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models/status')
def models_status():
    """Check status of loaded models"""
    status = {}
    
    for dataset, models in unified_predictor.models.items():
        status[dataset] = {
            'loaded_models': list(models.keys()),
            'total_models': len(models)
        }
    
    return jsonify(status)

@app.errorhandler(404)
def not_found(error):
    return render_template('base.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('base.html'), 500

if __name__ == '__main__':
    print("Starting Unified Diabetes Prediction Web Application...")
    print("="*60)
    print("🏥 Unified Diabetes Prediction System")
    print("📊 PIMA Models:", len(unified_predictor.models.get('PIMA', {})))
    print("📊 DDFH Models:", len(unified_predictor.models.get('DDFH', {})))
    print("🌐 Web Interface: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)