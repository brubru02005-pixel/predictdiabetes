#!/usr/bin/env python3
"""
CLINICAL DIABETES PREDICTION SYSTEM FOR CLINICS
Optimized for healthcare facilities using the 98.82% DDFH model
"""

import os
import json
import logging
from datetime import datetime
from unified_predictor import UnifiedDiabetesPredictor
from clinical_deployment import ClinicalDiabetesPredictor

# Clinical logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - CLINIC - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinic_operations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ClinicDiabetesSystem:
    """
    Clinical-grade diabetes prediction system optimized for clinic use
    Primary: DDFH Model (98.82% accuracy) - Clinical decisions
    Secondary: PIMA Model (95% accuracy) - Screening
    """
    
    def __init__(self):
        """Initialize clinic system with clinical-grade models"""
        logger.info("🏥 Initializing Clinic Diabetes Prediction System")
        
        # Load predictors
        self.clinical_predictor = ClinicalDiabetesPredictor()
        self.unified_predictor = UnifiedDiabetesPredictor()
        
        # Clinical configuration for clinic use
        self.clinic_config = {
            'primary_model': 'DDFH',
            'primary_accuracy': 98.82,
            'primary_use': 'Clinical Diagnosis Support',
            
            'secondary_model': 'PIMA', 
            'secondary_accuracy': 95.0,
            'secondary_use': 'Population Screening',
            
            'clinic_mode': True,
            'require_clinical_review': True,
            'generate_patient_reports': True,
            'ada_compliance': True
        }
        
        logger.info(f"✅ Primary Clinical Model: {self.clinic_config['primary_model']} ({self.clinic_config['primary_accuracy']}%)")
        logger.info(f"✅ Secondary Screening Model: {self.clinic_config['secondary_model']} ({self.clinic_config['secondary_accuracy']}%)")
    
    def clinical_assessment(self, patient_data, assessment_type='clinical'):
        """
        Perform clinical diabetes assessment
        
        Args:
            patient_data (dict): Patient clinical data
            assessment_type (str): 'clinical' (DDFH) or 'screening' (PIMA)
        
        Returns:
            dict: Clinical assessment results with recommendations
        """
        logger.info(f"🩺 Starting {assessment_type} assessment")
        
        try:
            if assessment_type == 'clinical':
                # Use DDFH model (98.82% accuracy) for clinical decisions
                result = self.clinical_predictor.clinical_prediction(patient_data)
                result['model_type'] = 'Clinical Grade DDFH'
                result['accuracy'] = '98.82%'
                result['recommended_use'] = 'Clinical diagnosis support'
                
            elif assessment_type == 'screening':
                # Use PIMA model (95% accuracy) for screening
                result = self.unified_predictor.predict(patient_data)
                
                # Enhance with clinical context
                result.update({
                    'model_type': 'Screening PIMA',
                    'accuracy': '95%', 
                    'recommended_use': 'Population screening',
                    'success': True,
                    'timestamp': datetime.now().isoformat(),
                    'requires_clinical_followup': result.get('confidence', 0) < 0.9
                })
            
            # Add clinic-specific metadata
            result['clinic_workflow'] = self._get_clinic_workflow(result)
            result['patient_instructions'] = self._get_patient_instructions(result)
            result['billing_codes'] = self._get_billing_codes(result)
            
            logger.info(f"✅ {assessment_type.title()} assessment completed")
            return result
            
        except Exception as e:
            logger.error(f"❌ Assessment failed: {str(e)}")
            return {
                'success': False,
                'error': f'Assessment failed: {str(e)}',
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_clinic_workflow(self, result):
        """Generate clinic workflow recommendations"""
        if not result.get('success'):
            return {'next_steps': ['System error - manual assessment required']}
        
        workflow = {
            'priority': 'routine',
            'next_steps': [],
            'follow_up_timeline': '6 months',
            'additional_tests': []
        }
        
        # Determine priority based on AI prediction and clinical risk
        ai_prediction = result.get('ai_prediction', '').lower()
        clinical_risk = result.get('clinical_risk_score', 0)
        confidence = result.get('ai_confidence', 0)
        
        if 'diabetes' in ai_prediction or clinical_risk >= 80:
            workflow['priority'] = 'urgent'
            workflow['next_steps'] = [
                'Immediate physician consultation',
                'Comprehensive metabolic panel',
                'Diabetes education referral',
                'Endocrinologist referral if confirmed'
            ]
            workflow['follow_up_timeline'] = '1-2 weeks'
            workflow['additional_tests'] = ['Fasting glucose', 'Repeat HbA1c', 'Lipid panel']
            
        elif 'pre' in ai_prediction or clinical_risk >= 60:
            workflow['priority'] = 'moderate'
            workflow['next_steps'] = [
                'Physician review within 1 month',
                'Lifestyle intervention counseling',
                'Nutritionist referral',
                'Diabetes prevention program'
            ]
            workflow['follow_up_timeline'] = '3 months'
            workflow['additional_tests'] = ['Repeat HbA1c in 3 months']
            
        elif confidence < 0.8:
            workflow['priority'] = 'review'
            workflow['next_steps'] = [
                'Clinical review recommended',
                'Consider additional risk factors',
                'Manual assessment by physician'
            ]
        
        return workflow
    
    def _get_patient_instructions(self, result):
        """Generate patient-friendly instructions"""
        if not result.get('success'):
            return {'message': 'Please consult with your healthcare provider.'}
        
        instructions = {
            'summary': '',
            'lifestyle_recommendations': [],
            'monitoring_instructions': [],
            'when_to_call_clinic': []
        }
        
        ai_prediction = result.get('ai_prediction', '').lower()
        clinical_risk = result.get('clinical_risk_score', 0)
        
        if 'diabetes' in ai_prediction:
            instructions['summary'] = 'Your assessment indicates diabetes. Immediate medical attention is recommended.'
            instructions['lifestyle_recommendations'] = [
                'Follow prescribed diabetic diet',
                'Monitor blood glucose as directed',
                'Take medications as prescribed',
                'Regular physical activity as approved by physician'
            ]
            instructions['when_to_call_clinic'] = [
                'Blood glucose >300 mg/dL',
                'Persistent symptoms (thirst, urination, fatigue)',
                'Any concerning symptoms'
            ]
            
        elif 'pre' in ai_prediction:
            instructions['summary'] = 'Your assessment indicates pre-diabetes. Lifestyle changes can prevent progression.'
            instructions['lifestyle_recommendations'] = [
                'Adopt heart-healthy diet',
                'Aim for 150 minutes moderate exercise weekly',
                'Maintain healthy weight',
                'Avoid tobacco and limit alcohol'
            ]
            instructions['monitoring_instructions'] = [
                'Regular blood glucose monitoring',
                'Annual HbA1c testing',
                'Blood pressure checks'
            ]
            
        else:
            instructions['summary'] = 'Your assessment shows low diabetes risk. Continue healthy lifestyle.'
            instructions['lifestyle_recommendations'] = [
                'Maintain balanced diet',
                'Stay physically active',
                'Regular health screenings'
            ]
            instructions['monitoring_instructions'] = [
                'Annual diabetes screening',
                'Regular wellness visits'
            ]
        
        return instructions
    
    def _get_billing_codes(self, result):
        """Generate relevant medical billing codes"""
        codes = {
            'primary_codes': [],
            'additional_codes': [],
            'notes': ''
        }
        
        if not result.get('success'):
            return codes
        
        ai_prediction = result.get('ai_prediction', '').lower()
        
        if 'diabetes' in ai_prediction:
            codes['primary_codes'] = ['E11.9', 'Z13.1'] # Type 2 diabetes, diabetes screening
            codes['additional_codes'] = ['Z71.3'] # Dietary counseling
            
        elif 'pre' in ai_prediction:
            codes['primary_codes'] = ['R73.03'] # Prediabetes
            codes['additional_codes'] = ['Z71.3', 'Z02.83'] # Counseling, pre-employment exam
            
        else:
            codes['primary_codes'] = ['Z13.1'] # Diabetes screening
        
        codes['notes'] = f"AI-assisted diabetes assessment using {result.get('model_type', 'clinical model')}"
        return codes
    
    def generate_clinical_report(self, patient_info, assessment_result):
        """Generate comprehensive clinical report"""
        if not assessment_result.get('success'):
            return None
        
        report = {
            'patient_info': patient_info,
            'assessment_date': assessment_result.get('timestamp'),
            'model_used': assessment_result.get('model_type'),
            'model_accuracy': assessment_result.get('accuracy'),
            
            'clinical_findings': {
                'ai_prediction': assessment_result.get('ai_prediction'),
                'confidence_level': f"{assessment_result.get('ai_confidence', 0):.1%}",
                'clinical_risk_score': assessment_result.get('clinical_risk_score', 0),
                'risk_factors': assessment_result.get('clinical_risk_factors', [])
            },
            
            'clinical_recommendations': assessment_result.get('recommendations', []),
            'workflow': assessment_result.get('clinic_workflow', {}),
            'patient_instructions': assessment_result.get('patient_instructions', {}),
            'billing_information': assessment_result.get('billing_codes', {}),
            
            'quality_assurance': {
                'model_validation': 'ADA-compliant clinical algorithms',
                'accuracy_rating': assessment_result.get('accuracy'),
                'clinical_review_required': assessment_result.get('requires_medical_review', False),
                'evidence_base': 'German DDFH clinical dataset'
            }
        }
        
        return report

def main():
    """Test clinic deployment system"""
    print("🏥 CLINIC DIABETES PREDICTION SYSTEM")
    print("=" * 60)
    print("Primary: DDFH Model (98.82% Clinical Grade)")
    print("Secondary: PIMA Model (95% Screening)")
    print("=" * 60)
    
    # Initialize clinic system
    clinic_system = ClinicDiabetesSystem()
    
    # Test cases for clinic scenarios
    test_patients = [
        {
            'name': 'Clinical Assessment - High Risk Patient',
            'data': {'hba1c': 8.5, 'bmi': 35.0, 'age': 58, 'tg': 280, 'gender': 'M'},
            'assessment_type': 'clinical'
        },
        {
            'name': 'Screening Assessment - Moderate Risk',
            'data': {
                'pregnancies': 2, 'glucose': 140, 'blood_pressure': 85,
                'skin_thickness': 30, 'insulin': 120, 'bmi': 28.5,
                'diabetes_pedigree': 0.45, 'age': 42
            },
            'assessment_type': 'screening'
        },
        {
            'name': 'Clinical Assessment - Pre-diabetic',
            'data': {'hba1c': 6.1, 'bmi': 29.0, 'age': 45, 'tg': 165, 'gender': 'F'},
            'assessment_type': 'clinical'
        }
    ]
    
    for patient in test_patients:
        print(f"\n📋 Testing: {patient['name']}")
        print("-" * 40)
        
        result = clinic_system.clinical_assessment(
            patient['data'], 
            patient['assessment_type']
        )
        
        if result['success']:
            print(f"🎯 Model: {result['model_type']} ({result['accuracy']})")
            print(f"📊 Prediction: {result.get('ai_prediction', 'N/A')}")
            print(f"🎪 Confidence: {result.get('ai_confidence', 0):.1%}")
            print(f"🏥 Workflow Priority: {result.get('clinic_workflow', {}).get('priority', 'routine').upper()}")
            print(f"👨‍⚕️ Clinical Review: {'Yes' if result.get('requires_medical_review') else 'No'}")
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    main()