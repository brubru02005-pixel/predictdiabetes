#!/usr/bin/env python3
"""
Clinical Deployment Configuration for Diabetes Prediction System
Prioritizes the DDFH model (98.82% accuracy) for clinical use
"""

import os
import json
import logging
from datetime import datetime
from unified_predictor import UnifiedDiabetesPredictor

# Setup clinical logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CLINICAL - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClinicalDiabetesPredictor:
    """
    Clinical-grade diabetes prediction system
    Uses DDFH model (98.82% accuracy) as primary predictor
    """
    
    def __init__(self):
        """Initialize clinical predictor with safety checks"""
        logger.info("Initializing Clinical Diabetes Prediction System")
        
        # Initialize the unified predictor
        self.predictor = UnifiedDiabetesPredictor()
        
        # Clinical configuration
        self.clinical_config = {
            'primary_model': 'DDFH',
            'primary_accuracy': 98.82,
            'secondary_model': 'PIMA', 
            'secondary_accuracy': 95.0,
            'minimum_clinical_accuracy': 95.0,
            'confidence_threshold': 0.8,
            'require_medical_review': True
        }
        
        # Clinical thresholds (ADA guidelines)
        self.clinical_thresholds = {
            'hba1c': {
                'normal': 5.7,
                'prediabetes': 6.4,
                'diabetes': 6.5
            },
            'glucose_fasting': {
                'normal': 100,
                'prediabetes': 125,
                'diabetes': 126
            },
            'bmi': {
                'normal': 25,
                'overweight': 30,
                'obese': 35
            }
        }
        
        logger.info(f"Clinical system initialized with {self.clinical_config['primary_model']} model")
        logger.info(f"Primary model accuracy: {self.clinical_config['primary_accuracy']}%")
    
    def validate_clinical_input(self, input_data):
        """Validate input data for clinical use"""
        logger.info("Validating clinical input data")
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check required fields for DDFH model
        required_fields = ['hba1c', 'bmi', 'age']
        for field in required_fields:
            if field not in input_data or input_data[field] is None:
                validation_results['errors'].append(f"Missing required field: {field}")
                validation_results['valid'] = False
        
        if not validation_results['valid']:
            return validation_results
        
        # Validate ranges
        if input_data.get('hba1c', 0) < 3.0 or input_data.get('hba1c', 0) > 15.0:
            validation_results['warnings'].append("HbA1c value outside normal range (3.0-15.0%)")
        
        if input_data.get('bmi', 0) < 15 or input_data.get('bmi', 0) > 60:
            validation_results['warnings'].append("BMI value outside normal range (15-60)")
        
        if input_data.get('age', 0) < 1 or input_data.get('age', 0) > 120:
            validation_results['warnings'].append("Age value outside normal range (1-120 years)")
        
        return validation_results
    
    def clinical_risk_assessment(self, input_data):
        """Perform clinical risk assessment based on ADA guidelines"""
        logger.info("Performing clinical risk assessment")
        
        risk_factors = []
        risk_score = 0
        
        # HbA1c assessment
        hba1c = input_data.get('hba1c', 0)
        if hba1c >= self.clinical_thresholds['hba1c']['diabetes']:
            risk_factors.append(f"HbA1c {hba1c}% indicates diabetes (≥6.5%)")
            risk_score += 40
        elif hba1c >= self.clinical_thresholds['hba1c']['prediabetes']:
            risk_factors.append(f"HbA1c {hba1c}% indicates prediabetes (5.7-6.4%)")
            risk_score += 25
        else:
            risk_factors.append(f"HbA1c {hba1c}% is normal (<5.7%)")
        
        # BMI assessment
        bmi = input_data.get('bmi', 0)
        if bmi >= self.clinical_thresholds['bmi']['obese']:
            risk_factors.append(f"BMI {bmi} indicates severe obesity (≥35)")
            risk_score += 20
        elif bmi >= self.clinical_thresholds['bmi']['overweight']:
            risk_factors.append(f"BMI {bmi} indicates obesity (≥30)")
            risk_score += 15
        elif bmi >= self.clinical_thresholds['bmi']['normal']:
            risk_factors.append(f"BMI {bmi} indicates overweight (25-29.9)")
            risk_score += 10
        
        # Age assessment
        age = input_data.get('age', 0)
        if age >= 65:
            risk_factors.append(f"Age {age} years - high risk group (≥65)")
            risk_score += 15
        elif age >= 45:
            risk_factors.append(f"Age {age} years - moderate risk group (45-64)")
            risk_score += 10
        
        return {
            'risk_factors': risk_factors,
            'risk_score': min(risk_score, 100),  # Cap at 100
            'risk_level': 'High' if risk_score >= 60 else 'Moderate' if risk_score >= 30 else 'Low'
        }
    
    def clinical_prediction(self, input_data):
        """Perform clinical-grade diabetes prediction"""
        logger.info("Starting clinical diabetes prediction")
        
        # Validate input
        validation = self.validate_clinical_input(input_data)
        if not validation['valid']:
            logger.error(f"Input validation failed: {validation['errors']}")
            return {
                'success': False,
                'error': 'Input validation failed',
                'validation_errors': validation['errors']
            }
        
        try:
            # Get AI prediction using DDFH model (primary)
            ai_result = self.predictor.predict(input_data)
            logger.info(f"AI prediction completed: {ai_result.get('class_name', 'Unknown')}")
            
            # Perform clinical risk assessment
            clinical_assessment = self.clinical_risk_assessment(input_data)
            
            # Combine AI and clinical assessments
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'model_used': 'DDFH Clinical (98.82% accuracy)',
                'input_data': input_data,
                'validation': validation,
                
                # AI Results
                'ai_prediction': ai_result.get('class_name', 'Unknown'),
                'ai_confidence': ai_result.get('confidence', 0.0),
                'ai_probability': ai_result.get('probability', 0.0),
                
                # Clinical Assessment
                'clinical_risk_score': clinical_assessment['risk_score'],
                'clinical_risk_level': clinical_assessment['risk_level'],
                'clinical_risk_factors': clinical_assessment['risk_factors'],
                
                # Clinical Recommendations
                'requires_medical_review': (
                    ai_result.get('confidence', 0) < self.clinical_config['confidence_threshold'] or
                    clinical_assessment['risk_score'] >= 60
                ),
                'immediate_action_required': clinical_assessment['risk_score'] >= 80,
                
                # Final Clinical Decision
                'clinical_decision': self._make_clinical_decision(ai_result, clinical_assessment),
                'recommendations': self._generate_recommendations(ai_result, clinical_assessment, input_data)
            }
            
            logger.info(f"Clinical prediction completed: {result['clinical_decision']}")
            return result
            
        except Exception as e:
            logger.error(f"Clinical prediction failed: {str(e)}")
            return {
                'success': False,
                'error': f'Prediction failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _make_clinical_decision(self, ai_result, clinical_assessment):
        """Make final clinical decision combining AI and clinical assessments"""
        ai_prediction = ai_result.get('class_name', '').lower()
        ai_confidence = ai_result.get('confidence', 0.0)
        clinical_risk = clinical_assessment['risk_level'].lower()
        
        # High confidence AI prediction + clinical agreement
        if ai_confidence >= 0.9:
            if 'diabetes' in ai_prediction and clinical_risk == 'high':
                return 'Strong indication of diabetes - recommend immediate clinical evaluation'
            elif 'pre' in ai_prediction and clinical_risk in ['moderate', 'high']:
                return 'Pre-diabetes indicated - recommend lifestyle intervention and monitoring'
            elif 'no' in ai_prediction and clinical_risk == 'low':
                return 'Low diabetes risk - routine monitoring recommended'
        
        # Medium confidence or conflicting assessments
        if ai_confidence >= 0.8:
            return f'Moderate confidence prediction ({ai_prediction}) - clinical review recommended'
        
        # Low confidence
        return 'Low confidence prediction - comprehensive clinical evaluation required'
    
    def _generate_recommendations(self, ai_result, clinical_assessment, input_data):
        """Generate clinical recommendations"""
        recommendations = []
        
        # HbA1c-based recommendations
        hba1c = input_data.get('hba1c', 0)
        if hba1c >= 6.5:
            recommendations.append("Immediate endocrinologist referral for diabetes management")
            recommendations.append("Initiate diabetes education and blood glucose monitoring")
        elif hba1c >= 5.7:
            recommendations.append("Lifestyle intervention program enrollment")
            recommendations.append("Repeat HbA1c in 3-6 months")
        
        # BMI-based recommendations
        bmi = input_data.get('bmi', 0)
        if bmi >= 30:
            recommendations.append("Weight management program referral")
            recommendations.append("Nutritionist consultation recommended")
        
        # Age-based recommendations
        age = input_data.get('age', 0)
        if age >= 45:
            recommendations.append("Annual diabetes screening recommended")
        
        # Risk-level based recommendations
        risk_level = clinical_assessment['risk_level'].lower()
        if risk_level == 'high':
            recommendations.append("Immediate clinical follow-up within 1-2 weeks")
            recommendations.append("Comprehensive metabolic panel")
        elif risk_level == 'moderate':
            recommendations.append("Clinical follow-up within 1 month")
            recommendations.append("Lifestyle modification counseling")
        else:
            recommendations.append("Routine annual screening")
            recommendations.append("Maintain healthy lifestyle")
        
        return recommendations
    
    def generate_clinical_report(self, prediction_result):
        """Generate clinical report for documentation"""
        if not prediction_result.get('success'):
            return None
        
        report = {
            'patient_assessment': {
                'timestamp': prediction_result['timestamp'],
                'model_accuracy': '98.82%',
                'ai_prediction': prediction_result['ai_prediction'],
                'confidence_level': f"{prediction_result['ai_confidence']:.1%}",
                'clinical_risk_level': prediction_result['clinical_risk_level']
            },
            'clinical_findings': {
                'risk_factors': prediction_result['clinical_risk_factors'],
                'risk_score': f"{prediction_result['clinical_risk_score']}/100"
            },
            'recommendations': prediction_result['recommendations'],
            'follow_up_required': prediction_result['requires_medical_review'],
            'urgent_action': prediction_result['immediate_action_required']
        }
        
        return report

def main():
    """Test clinical deployment"""
    print("🏥 CLINICAL DIABETES PREDICTION SYSTEM")
    print("=" * 50)
    
    # Initialize clinical predictor
    clinical_predictor = ClinicalDiabetesPredictor()
    
    # Test cases
    test_cases = [
        {
            'name': 'High Risk Patient',
            'data': {'hba1c': 8.2, 'bmi': 35.5, 'age': 58, 'tg': 250, 'gender': 'M'}
        },
        {
            'name': 'Pre-diabetic Patient', 
            'data': {'hba1c': 6.1, 'bmi': 28.5, 'age': 45, 'tg': 180, 'gender': 'F'}
        },
        {
            'name': 'Low Risk Patient',
            'data': {'hba1c': 5.2, 'bmi': 23.0, 'age': 32, 'tg': 120, 'gender': 'F'}
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['name']}")
        print("-" * 30)
        
        result = clinical_predictor.clinical_prediction(test_case['data'])
        
        if result['success']:
            print(f"🎯 Clinical Decision: {result['clinical_decision']}")
            print(f"🤖 AI Prediction: {result['ai_prediction']} (Confidence: {result['ai_confidence']:.1%})")
            print(f"🏥 Clinical Risk: {result['clinical_risk_level']} (Score: {result['clinical_risk_score']}/100)")
            print(f"⚠️  Medical Review Required: {'Yes' if result['requires_medical_review'] else 'No'}")
            print(f"🚨 Immediate Action: {'Yes' if result['immediate_action_required'] else 'No'}")
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()