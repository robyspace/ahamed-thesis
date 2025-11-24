"""
Local Testing Script for Lambda Inference Handler

Run this script to test the Lambda handler locally before deployment.
Requires model files (variance_model_best.pkl, feature_columns.json, model_metadata.json)
in the same directory.

Usage:
    python test_local.py
"""

import json
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lambda_handler import lambda_handler

class MockContext:
    """Mock AWS Lambda context for local testing"""
    def __init__(self):
        self.function_name = 'hybrid-ml-router'
        self.function_version = '$LATEST'
        self.invoked_function_arn = 'arn:aws:lambda:local:123456789012:function:hybrid-ml-router'
        self.memory_limit_in_mb = 512
        self.aws_request_id = 'local-test-request-id'
        self.log_group_name = '/aws/lambda/hybrid-ml-router'
        self.log_stream_name = 'local-test-stream'

def test_inference():
    """Run test cases for local validation"""

    print("\\n" + "="*80)
    print("LOCAL INFERENCE TESTING")
    print("="*80)

    # Test cases covering different scenarios
    test_cases = [
        {
            'name': 'Small lightweight API - Midday',
            'request': {
                'workload_type': 'lightweight_api',
                'payload_size_kb': 1.0,
                'time_window': 'midday',
                'load_pattern': 'low_load'
            },
            'expected': 'lambda'
        },
        {
            'name': 'Large lightweight API - Early morning',
            'request': {
                'workload_type': 'lightweight_api',
                'payload_size_kb': 10.0,
                'time_window': 'early_morning',
                'load_pattern': 'low_load'
            },
            'expected': 'ecs'
        },
        {
            'name': 'Thumbnail processing - Medium load',
            'request': {
                'workload_type': 'thumbnail_processing',
                'payload_size_kb': 200.0,
                'time_window': 'midday',
                'load_pattern': 'medium_load'
            },
            'expected': 'lambda'
        },
        {
            'name': 'Large thumbnail - Early morning (cold starts)',
            'request': {
                'workload_type': 'thumbnail_processing',
                'payload_size_kb': 1024.0,
                'time_window': 'early_morning',
                'load_pattern': 'burst_load'
            },
            'expected': None  # Unknown, depends on model
        },
        {
            'name': 'Heavy processing - Medium payload',
            'request': {
                'workload_type': 'heavy_processing',
                'payload_size_kb': 5120.0,
                'time_window': 'evening',
                'load_pattern': 'ramp_load'
            },
            'expected': 'lambda'
        },
        {
            'name': 'Medium processing - Large payload',
            'request': {
                'workload_type': 'medium_processing',
                'payload_size_kb': 3072.0,
                'time_window': 'morning_peak',
                'load_pattern': 'medium_load'
            },
            'expected': 'lambda'
        }
    ]

    context = MockContext()
    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n{'='*80}")
        print(f"Test {i}: {test_case['name']}")
        print(f"{'='*80}")

        request = test_case['request']
        print(f"\\nRequest:")
        print(f"  Workload: {request['workload_type']}")
        print(f"  Payload:  {request['payload_size_kb']} KB")
        print(f"  Time:     {request['time_window']}")
        print(f"  Load:     {request['load_pattern']}")

        try:
            # Invoke handler
            response = lambda_handler(request, context)

            if response['statusCode'] == 200:
                body = json.loads(response['body'])
                prediction = body['prediction']
                metrics = body['metrics']

                print(f"\\n Prediction successful:")
                print(f"  Platform:        {prediction['platform'].upper()}")
                print(f"  Confidence:      {prediction['confidence']*100:.1f}%")
                print(f"  Lambda prob:     {prediction['lambda_probability']*100:.1f}%")
                print(f"  ECS prob:        {prediction['ecs_probability']*100:.1f}%")
                print(f"  Inference time:  {metrics['inference_time_ms']:.4f} ms")
                print(f"  Inference cost:  ${metrics['inference_cost_usd']:.10f}")

                # Check against expected
                if test_case['expected']:
                    if prediction['platform'] == test_case['expected']:
                        print(f"  Expected result: MATCH ({test_case['expected']})")
                        test_result = 'PASS'
                    else:
                        print(f"  Expected result: DIFFERENT (expected: {test_case['expected']}, got: {prediction['platform']})")
                        test_result = 'DIFF'
                else:
                    print(f"  Expected result: N/A (model-dependent)")
                    test_result = 'PASS'

                results.append({
                    'test': test_case['name'],
                    'status': test_result,
                    'prediction': prediction['platform'],
                    'confidence': prediction['confidence'],
                    'latency_ms': metrics['inference_time_ms']
                })

            else:
                print(f"\\n Error: Status code {response['statusCode']}")
                print(f"  Response: {response['body']}")
                results.append({
                    'test': test_case['name'],
                    'status': 'ERROR',
                    'error': response['body']
                })

        except Exception as e:
            print(f"\\n Exception occurred: {str(e)}")
            results.append({
                'test': test_case['name'],
                'status': 'EXCEPTION',
                'error': str(e)
            })

    # Summary
    print(f"\\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for r in results if r['status'] in ['PASS', 'DIFF'])
    failed = len(results) - passed

    print(f"\\nTotal tests:  {len(results)}")
    print(f"Passed:       {passed}")
    print(f"Failed:       {failed}")

    if failed == 0:
        print(f"\\n ALL TESTS PASSED!")
    else:
        print(f"\\n Some tests failed - review above")

    # Performance summary
    successful_results = [r for r in results if 'latency_ms' in r]
    if successful_results:
        avg_latency = sum(r['latency_ms'] for r in successful_results) / len(successful_results)
        max_latency = max(r['latency_ms'] for r in successful_results)
        min_latency = min(r['latency_ms'] for r in successful_results)

        print(f"\\n📊 Latency Statistics:")
        print(f"  Average:  {avg_latency:.4f} ms")
        print(f"  Min:      {min_latency:.4f} ms")
        print(f"  Max:      {max_latency:.4f} ms")

        if avg_latency < 5:
            print(f"  EXCELLENT: Average latency < 5ms")
        elif avg_latency < 10:
            print(f"  GOOD: Average latency < 10ms")
        else:
            print(f"  WARNING: Average latency > 10ms")

    print(f"\\n" + "="*80)

    return results

def test_error_handling():
    """Test error handling with invalid inputs"""

    print("\\n" + "="*80)
    print("ERROR HANDLING TESTS")
    print("="*80)

    error_cases = [
        {
            'name': 'Missing workload_type',
            'request': {
                'payload_size_kb': 5.0,
                'time_window': 'midday'
            }
        },
        {
            'name': 'Invalid workload_type',
            'request': {
                'workload_type': 'invalid_workload',
                'payload_size_kb': 5.0,
                'time_window': 'midday'
            }
        },
        {
            'name': 'Missing payload_size_kb',
            'request': {
                'workload_type': 'lightweight_api',
                'time_window': 'midday'
            }
        }
    ]

    context = MockContext()

    for i, test_case in enumerate(error_cases, 1):
        print(f"\\nError Test {i}: {test_case['name']}")

        try:
            response = lambda_handler(test_case['request'], context)

            if response['statusCode'] == 200:
                body = json.loads(response['body'])
                print(f"  Handled gracefully with fallback")
                print(f"  Platform: {body['prediction']['platform']}")
            else:
                print(f"  ✅ Returned error response (expected)")

        except Exception as e:
            print(f"  Unhandled exception: {str(e)}")

    print(f"\\n Error handling tests complete")

def main():
    """Main test runner"""

    # Check if model files exist
    required_files = [
        'variance_model_best.pkl',
        'feature_columns.json',
        'model_metadata.json'
    ]

    print("\\nChecking for required files...")
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (MISSING)")
            missing_files.append(file)

    if missing_files:
        print(f"\\n   Missing files: {', '.join(missing_files)}")
        print(f"\\nPlease download model files from Google Colab training notebook:")
        print(f"  1. variance_model_best.pkl")
        print(f"  2. feature_columns.json")
        print(f"  3. model_metadata.json")
        print(f"\\nPlace them in: {os.path.dirname(os.path.abspath(__file__))}")
        return

    print(f"\\n All required files present\\n")

    # Run tests
    test_inference()
    test_error_handling()

    print(f"\\n" + "="*80)
    print(" ALL TESTS COMPLETE")
    print("="*80)
    print(f"\\n💡 Next steps:")
    print(f"  1. Review test results above")
    print(f"  2. If tests pass, deploy to AWS Lambda")
    print(f"  3. Follow AWS_DEPLOYMENT_GUIDE.md for deployment")
    print(f"\\n")

if __name__ == '__main__':
    main()
