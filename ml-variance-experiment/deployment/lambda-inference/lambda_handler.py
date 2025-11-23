"""
AWS Lambda Handler for ML-Based Platform Routing
Hybrid Serverless-Container Thesis - Inference Module

This Lambda function receives workload requests and predicts the optimal
platform (Lambda vs ECS) using the trained ML model.

Features:
- Real-time inference (<10ms target)
- Cost and latency tracking
- Confidence scores
- Error handling and fallback logic
"""

import json
import numpy as np
import joblib
import time
from datetime import datetime

# Global variables (loaded once per container)
MODEL = None
FEATURE_COLUMNS = None
METADATA = None

# AWS Pricing Constants (eu-west-1)
LAMBDA_REQUEST_COST = 0.20 / 1_000_000
LAMBDA_DURATION_COST_PER_GB_SEC = 0.0000166667

def load_model():
    """Load model and metadata (called once per Lambda container)"""
    global MODEL, FEATURE_COLUMNS, METADATA

    if MODEL is None:
        print("Loading model...")
        MODEL = joblib.load('variance_model_best.pkl')

        with open('feature_columns.json', 'r') as f:
            FEATURE_COLUMNS = json.load(f)

        with open('model_metadata.json', 'r') as f:
            METADATA = json.load(f)

        print(f"Model loaded: {METADATA['model_type']}")
        print(f"Test accuracy: {METADATA['performance']['test_accuracy']:.4f}")

    return MODEL, FEATURE_COLUMNS, METADATA

def extract_features(request_data):
    """
    Extract features from incoming request.

    Expected request_data format:
    {
        "workload_type": "lightweight_api" | "thumbnail_processing" | "medium_processing" | "heavy_processing",
        "payload_size_kb": <float>,
        "time_window": "early_morning" | "morning_peak" | "midday" | "evening" | "late_night",
        "load_pattern": "low_load" | "medium_load" | "burst_load" | "ramp_load",
        "timestamp": <ISO timestamp> (optional)
    }
    """

    # Workload encoding
    workload_encoding = {
        'lightweight_api': 0,
        'thumbnail_processing': 1,
        'medium_processing': 2,
        'heavy_processing': 3
    }

    # Time window encoding
    time_window_encoding = {
        'early_morning': 0,
        'morning_peak': 1,
        'midday': 2,
        'evening': 3,
        'late_night': 4,
        'other': 5
    }

    # Load pattern encoding
    load_pattern_encoding = {
        'low_load': 0,
        'medium_load': 1,
        'burst_load': 2,
        'ramp_load': 3
    }

    # Lambda memory configs
    lambda_memory_configs = {
        'lightweight_api': 128,
        'thumbnail_processing': 512,
        'medium_processing': 1024,
        'heavy_processing': 2048
    }

    # Extract or derive values
    workload_type = request_data.get('workload_type', 'lightweight_api')
    payload_size_kb = float(request_data.get('payload_size_kb', 1.0))
    time_window = request_data.get('time_window', 'other')
    load_pattern = request_data.get('load_pattern', 'low_load')

    # Get timestamp or use current time
    if 'timestamp' in request_data:
        timestamp = datetime.fromisoformat(request_data['timestamp'].replace('Z', '+00:00'))
    else:
        timestamp = datetime.utcnow()

    hour_of_day = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0

    # Encode categorical features
    workload_type_encoded = workload_encoding.get(workload_type, 0)
    time_window_encoded = time_window_encoding.get(time_window, 5)
    load_pattern_encoded = load_pattern_encoding.get(load_pattern, 0)
    lambda_memory_limit_mb = lambda_memory_configs.get(workload_type, 128)

    # Engineered features
    payload_squared = payload_size_kb ** 2
    payload_log = np.log1p(payload_size_kb)
    payload_workload_interaction = payload_size_kb * workload_type_encoded
    payload_hour_interaction = payload_size_kb * hour_of_day
    payload_time_window_interaction = payload_size_kb * time_window_encoded
    workload_time_window_interaction = workload_type_encoded * time_window_encoded
    payload_load_pattern_interaction = payload_size_kb * load_pattern_encoded
    time_window_load_pattern_interaction = time_window_encoded * load_pattern_encoded

    # Create feature dictionary (must match FEATURE_COLUMNS order)
    features = {
        'workload_type_encoded': workload_type_encoded,
        'payload_size_kb': payload_size_kb,
        'time_window_encoded': time_window_encoded,
        'load_pattern_encoded': load_pattern_encoded,
        'hour_of_day': hour_of_day,
        'is_weekend': is_weekend,
        'lambda_memory_limit_mb': lambda_memory_limit_mb,
        'payload_squared': payload_squared,
        'payload_log': payload_log,
        'payload_workload_interaction': payload_workload_interaction,
        'payload_hour_interaction': payload_hour_interaction,
        'payload_time_window_interaction': payload_time_window_interaction,
        'workload_time_window_interaction': workload_time_window_interaction,
        'payload_load_pattern_interaction': payload_load_pattern_interaction,
        'time_window_load_pattern_interaction': time_window_load_pattern_interaction
    }

    return features

def predict_platform(features, model, feature_columns):
    """Make prediction using the trained model"""

    # Create feature array in correct order
    feature_array = np.array([features[col] for col in feature_columns]).reshape(1, -1)

    # Make prediction
    prediction = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]

    # Return results
    return {
        'platform': 'lambda' if prediction == 1 else 'ecs',
        'confidence': float(max(probabilities)),
        'lambda_probability': float(probabilities[1]),
        'ecs_probability': float(probabilities[0])
    }

def calculate_inference_cost(execution_time_ms, memory_mb=512):
    """Calculate the cost of this Lambda inference"""
    memory_gb = memory_mb / 1024
    execution_time_sec = execution_time_ms / 1000
    duration_cost = memory_gb * execution_time_sec * LAMBDA_DURATION_COST_PER_GB_SEC
    total_cost = duration_cost + LAMBDA_REQUEST_COST
    return total_cost

def lambda_handler(event, context):
    """
    AWS Lambda handler function

    Event format:
    {
        "workload_type": "lightweight_api",
        "payload_size_kb": 5.0,
        "time_window": "midday",
        "load_pattern": "medium_load",
        "timestamp": "2025-11-23T12:00:00Z"
    }

    Response format:
    {
        "statusCode": 200,
        "body": {
            "prediction": {
                "platform": "lambda" | "ecs",
                "confidence": 0.85,
                "lambda_probability": 0.85,
                "ecs_probability": 0.15
            },
            "metrics": {
                "inference_time_ms": 2.5,
                "inference_cost_usd": 0.0000001,
                "timestamp": "2025-11-23T12:00:00.123456"
            },
            "request_context": {
                "workload_type": "lightweight_api",
                "payload_size_kb": 5.0,
                "time_window": "midday"
            }
        }
    }
    """

    start_time = time.time()

    try:
        # Load model (cached after first call)
        model, feature_columns, metadata = load_model()

        # Parse request
        if isinstance(event, str):
            request_data = json.loads(event)
        elif 'body' in event:
            request_data = json.loads(event['body']) if isinstance(event['body'], str) else event['body']
        else:
            request_data = event

        # Extract features
        features = extract_features(request_data)

        # Make prediction
        prediction_result = predict_platform(features, model, feature_columns)

        # Calculate metrics
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        inference_cost = calculate_inference_cost(
            inference_time_ms,
            memory_mb=context.memory_limit_in_mb if context else 512
        )

        # Build response
        response = {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': prediction_result,
                'metrics': {
                    'inference_time_ms': round(inference_time_ms, 4),
                    'inference_cost_usd': inference_cost,
                    'timestamp': datetime.utcnow().isoformat() + 'Z'
                },
                'request_context': {
                    'workload_type': request_data.get('workload_type'),
                    'payload_size_kb': request_data.get('payload_size_kb'),
                    'time_window': request_data.get('time_window'),
                    'load_pattern': request_data.get('load_pattern')
                },
                'model_info': {
                    'model_type': metadata['model_type'],
                    'test_accuracy': metadata['performance']['test_accuracy'],
                    'training_date': metadata['training_date']
                }
            })
        }

        return response

    except Exception as e:
        # Error handling with fallback
        print(f"Error during inference: {str(e)}")

        # Fallback: simple heuristic routing
        fallback_platform = 'lambda'  # Default to Lambda for unknown cases

        if 'workload_type' in request_data:
            workload = request_data['workload_type']
            payload = request_data.get('payload_size_kb', 1.0)

            # Simple fallback logic
            if workload == 'lightweight_api' and payload > 7:
                fallback_platform = 'ecs'
            elif workload == 'heavy_processing' and payload > 8000:
                fallback_platform = 'ecs'

        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': str(e),
                'fallback_prediction': {
                    'platform': fallback_platform,
                    'confidence': 0.5,
                    'note': 'Fallback prediction due to error'
                }
            })
        }

# Local testing
if __name__ == '__main__':
    # Test cases
    test_requests = [
        {
            'workload_type': 'lightweight_api',
            'payload_size_kb': 1.0,
            'time_window': 'midday',
            'load_pattern': 'low_load'
        },
        {
            'workload_type': 'lightweight_api',
            'payload_size_kb': 10.0,
            'time_window': 'early_morning',
            'load_pattern': 'low_load'
        },
        {
            'workload_type': 'thumbnail_processing',
            'payload_size_kb': 200.0,
            'time_window': 'midday',
            'load_pattern': 'medium_load'
        },
        {
            'workload_type': 'heavy_processing',
            'payload_size_kb': 5120.0,
            'time_window': 'evening',
            'load_pattern': 'burst_load'
        }
    ]

    print("\\n" + "="*80)
    print("TESTING LAMBDA INFERENCE LOCALLY")
    print("="*80)

    # Mock context
    class MockContext:
        memory_limit_in_mb = 512

    for i, request in enumerate(test_requests, 1):
        print(f"\\nTest {i}: {request['workload_type']}, {request['payload_size_kb']} KB")

        result = lambda_handler(request, MockContext())

        if result['statusCode'] == 200:
            body = json.loads(result['body'])
            pred = body['prediction']
            metrics = body['metrics']

            print(f"  Prediction: {pred['platform'].upper()}")
            print(f"  Confidence: {pred['confidence']*100:.1f}%")
            print(f"  Inference time: {metrics['inference_time_ms']:.4f} ms")
            print(f"  Inference cost: ${metrics['inference_cost_usd']:.10f}")
        else:
            print(f"  Error: {result['body']}")

    print("\\n" + "="*80)
    print("✅ LOCAL TESTING COMPLETE")
    print("="*80)
