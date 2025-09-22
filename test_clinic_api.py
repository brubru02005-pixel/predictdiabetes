#!/usr/bin/env python3
"""
API TEST SCRIPT FOR CLINIC DEPLOYMENT SYSTEM
Tests the clinical-focused API endpoints
"""

import requests
import json
from datetime import datetime

def test_clinic_api():
    """Test the clinic deployment API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🏥 TESTING CLINIC DEPLOYMENT API")
    print("=" * 50)
    
    # Test cases for different clinical scenarios
    test_cases = [
        {
            'name': 'Clinical Assessment - High Risk',
            'endpoint': '/api/predict',
            'data': {
                'type': 'clinical',
                'hba1c': 8.5,
                'bmi': 35.0,
                'age': 58,
                'tg': 280,
                'gender': 'M'
            },
            'expected_model': 'Clinical Grade DDFH'
        },
        {
            'name': 'Screening Assessment - Moderate Risk',
            'endpoint': '/api/predict',
            'data': {
                'type': 'screening',
                'pregnancies': 2,
                'glucose': 140,
                'blood_pressure': 85,
                'skin_thickness': 30,
                'insulin': 120,
                'bmi': 28.5,
                'diabetes_pedigree': 0.45,
                'age': 42
            },
            'expected_model': 'Screening PIMA'
        },
        {
            'name': 'Auto-Detect Clinical (HbA1c present)',
            'endpoint': '/api/predict',
            'data': {
                'type': 'auto',
                'hba1c': 6.1,
                'bmi': 29.0,
                'age': 45,
                'tg': 165,
                'gender': 'F'
            },
            'expected_model': 'Clinical Grade DDFH'
        },
        {
            'name': 'Auto-Detect Screening (PIMA features)',
            'endpoint': '/api/predict',
            'data': {
                'type': 'auto',
                'pregnancies': 1,
                'glucose': 95,
                'blood_pressure': 70,
                'skin_thickness': 20,
                'insulin': 85,
                'bmi': 24.5,
                'diabetes_pedigree': 0.25,
                'age': 30
            },
            'expected_model': 'Screening PIMA'
        }
    ]
    
    print(f"🌐 Testing API endpoints at {base_url}")
    print("Note: Make sure Flask app is running (python diabetes_app.py)")
    print("-" * 50)
    
    # Test model status endpoint first
    try:
        print("\n📊 Testing Model Status Endpoint...")
        response = requests.get(f"{base_url}/models/status", timeout=5)
        
        if response.status_code == 200:
            status = response.json()
            print("✅ Model Status Retrieved:")
            
            clinic_status = status.get('clinic_system', {})
            print(f"   Clinic System: {'Available' if clinic_status.get('available') else 'Not Available'}")
            print(f"   Primary Model: {clinic_status.get('primary_model', 'N/A')}")
            print(f"   Secondary Model: {clinic_status.get('secondary_model', 'N/A')}")
            print(f"   Deployment Mode: {clinic_status.get('deployment_mode', 'Unknown')}")
        else:
            print(f"⚠️ Model Status Error: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Flask app: {e}")
        print("   Please start the Flask app first: python diabetes_app.py")
        return
    
    # Test prediction endpoints
    results_summary = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 TEST {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(
                f"{base_url}{test_case['endpoint']}",
                json=test_case['data'],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success'):
                    # Extract key information
                    model_type = result.get('model_type', 'Unknown')
                    ai_prediction = result.get('ai_prediction', 'N/A')
                    confidence = result.get('ai_confidence', 0)
                    accuracy = result.get('accuracy', 'N/A')
                    
                    # Display results
                    print(f"✅ Success")
                    print(f"   Model: {model_type}")
                    print(f"   Prediction: {ai_prediction}")
                    print(f"   Confidence: {confidence:.1%}" if confidence else f"   Confidence: {confidence}")
                    print(f"   Model Accuracy: {accuracy}")
                    
                    # Check if expected model was used
                    expected = test_case.get('expected_model', '')
                    model_match = expected.lower() in model_type.lower() if expected else True
                    
                    # Show workflow info if available
                    workflow = result.get('clinic_workflow', {})
                    if workflow:
                        priority = workflow.get('priority', 'routine')
                        print(f"   Clinical Priority: {priority.upper()}")
                    
                    results_summary.append({
                        'test': test_case['name'],
                        'status': '✅ PASS' if model_match else '⚠️ WRONG MODEL',
                        'model': model_type,
                        'prediction': ai_prediction,
                        'expected': expected
                    })
                    
                else:
                    print(f"❌ Prediction Failed: {result.get('error', 'Unknown error')}")
                    results_summary.append({
                        'test': test_case['name'],
                        'status': '❌ FAIL',
                        'error': result.get('error', 'Unknown error')
                    })
                    
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text[:100]}...")
                
                results_summary.append({
                    'test': test_case['name'],
                    'status': '❌ HTTP ERROR',
                    'http_code': response.status_code
                })
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Failed: {str(e)}")
            results_summary.append({
                'test': test_case['name'],
                'status': '❌ REQUEST FAILED',
                'error': str(e)
            })
    
    # Summary Report
    print("\n" + "=" * 50)
    print("📊 CLINIC API TEST SUMMARY")
    print("=" * 50)
    
    for i, result in enumerate(results_summary, 1):
        status = result.get('status', 'Unknown')
        print(f"{i}. {result['test']}: {status}")
        
        if 'model' in result:
            print(f"   Model Used: {result.get('model', 'N/A')}")
            print(f"   Prediction: {result.get('prediction', 'N/A')}")
            if result.get('expected'):
                print(f"   Expected Model: {result.get('expected')}")
    
    # Test Summary
    total_tests = len(results_summary)
    passed_tests = len([r for r in results_summary if '✅' in r.get('status', '')])
    
    print(f"\n🎯 TEST RESULTS: {passed_tests}/{total_tests} PASSED")
    
    if passed_tests == total_tests:
        print("🏆 ALL TESTS PASSED - CLINIC API READY FOR DEPLOYMENT")
    else:
        print("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
    
    print("=" * 50)

if __name__ == "__main__":
    test_clinic_api()