from flask import Flask, request, jsonify
import time
import random
import base64
from datetime import datetime
import os
import psutil

app = Flask(__name__)

def process_workload():
    """Universal handler for all workload types"""
    start_time = time.time()
    
    # Parse request
    data = request.get_json() if request.is_json else {}
    workload_type = data.get('workload_type', 'lightweight_api')
    payload_data = data.get('payload', '')
    request_id = data.get('request_id', str(time.time()))
    
    # Execute workload
    result = execute_workload(workload_type, payload_data)
    
    # Calculate metrics
    execution_time = (time.time() - start_time) * 1000  # ms
    process = psutil.Process(os.getpid())
    memory_used = process.memory_info().rss / 1024 / 1024  # MB
    
    response = {
        'request_id': request_id,
        'platform': 'ecs',
        'workload_type': workload_type,
        'timestamp': datetime.utcnow().isoformat(),
        'metrics': {
            'execution_time_ms': execution_time,
            'memory_used_mb': memory_used,
            'cpu_percent': process.cpu_percent(interval=0.1),
            'payload_size_kb': len(str(payload_data)) / 1024
        },
        'result': result
    }
    
    return jsonify(response), 200

# Routes for each workload type (matching ALB path-based routing)
@app.route('/lightweight/process', methods=['POST'])
def lightweight_process():
    return process_workload()

@app.route('/thumbnail/process', methods=['POST'])
def thumbnail_process():
    return process_workload()

@app.route('/medium/process', methods=['POST'])
def medium_process():
    return process_workload()

@app.route('/heavy/process', methods=['POST'])
def heavy_process():
    return process_workload()

# Keep the original /process route for backward compatibility
@app.route('/process', methods=['POST'])
def process_request():
    return process_workload()

# Health check routes
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'hybrid-thesis-app'}), 200

@app.route('/lightweight/health', methods=['GET'])
def lightweight_health():
    return jsonify({'status': 'healthy', 'service': 'lightweight'}), 200

@app.route('/thumbnail/health', methods=['GET'])
def thumbnail_health():
    return jsonify({'status': 'healthy', 'service': 'thumbnail'}), 200

@app.route('/medium/health', methods=['GET'])
def medium_health():
    return jsonify({'status': 'healthy', 'service': 'medium'}), 200

@app.route('/heavy/health', methods=['GET'])
def heavy_health():
    return jsonify({'status': 'healthy', 'service': 'heavy'}), 200

def execute_workload(workload_type, payload):
    """Execute different workload types - identical to Lambda version"""
    if workload_type == 'lightweight_api':
        time.sleep(random.uniform(0.05, 0.15))
        return {'validated': True, 'size': len(str(payload))}
    
    elif workload_type == 'thumbnail_processing':
        data = base64.b64decode(payload) if payload else b''
        time.sleep(random.uniform(0.2, 0.4))
        return {'processed': True, 'output_size': len(data) // 2}
    
    elif workload_type == 'medium_processing':
        time.sleep(random.uniform(0.8, 1.5))
        return {'processed': True, 'operations': 1000}
    
    elif workload_type == 'heavy_processing':
        time.sleep(random.uniform(3, 5))
        return {'processed': True, 'operations': 10000}
    
    return {'status': 'unknown_workload'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)