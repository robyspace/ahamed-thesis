import json
import time
import random
import base64
from datetime import datetime
import os

# Global variable to detect cold starts
is_cold_start = True

def lambda_handler(event, context):
    """Universal handler for all workload types"""
    global is_cold_start
    
    start_time = time.time()
    cold_start = is_cold_start
    is_cold_start = False  # Subsequent invocations will be warm
    
    # Parse request
    body = json.loads(event.get('body', '{}'))
    workload_type = body.get('workload_type', 'lightweight_api')
    payload_data = body.get('payload', '')
    request_id = body.get('request_id', context.aws_request_id)  # FIX: aws_request_id not request_id
    
    # Execute workload
    result = execute_workload(workload_type, payload_data)
    
    # Calculate metrics
    execution_time = (time.time() - start_time) * 1000  # ms
    memory_used = context.memory_limit_in_mb
    
    response = {
        'request_id': request_id,
        'platform': 'lambda',
        'workload_type': workload_type,
        'timestamp': datetime.utcnow().isoformat(),
        'metrics': {
            'execution_time_ms': execution_time,
            'memory_used_mb': memory_used,
            'memory_limit_mb': context.memory_limit_in_mb,
            'cold_start': cold_start,  # FIX: Use proper cold start detection
            'payload_size_kb': len(json.dumps(payload_data)) / 1024
        },
        'result': result
    }
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'X-Platform': 'lambda',
            'X-Execution-Time': str(execution_time)
        },
        'body': json.dumps(response)
    }

def execute_workload(workload_type, payload):
    """Execute different workload types"""
    if workload_type == 'lightweight_api':
        # Simple validation logic
        time.sleep(random.uniform(0.05, 0.15))
        return {'validated': True, 'size': len(str(payload))}
    
    elif workload_type == 'thumbnail_processing':
        # Simulate image processing
        data = base64.b64decode(payload) if payload else b''
        time.sleep(random.uniform(0.2, 0.4))
        return {'processed': True, 'output_size': len(data) // 2}
    
    elif workload_type == 'medium_processing':
        # Medium computation
        time.sleep(random.uniform(0.8, 1.5))
        return {'processed': True, 'operations': 1000}
    
    elif workload_type == 'heavy_processing':
        # Heavy computation
        time.sleep(random.uniform(3, 5))
        return {'processed': True, 'operations': 10000}
    
    return {'status': 'unknown_workload'}