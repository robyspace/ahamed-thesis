#!/usr/bin/env python3
"""
Workload Generator for ML Router Evaluation
Generates realistic workload patterns to test routing strategies
"""

import random
import time
import json
from datetime import datetime
from typing import Dict, List

# Workload configurations with realistic payload sizes
WORKLOAD_CONFIGS = {
    'lightweight_api': {
        'payload_sizes_kb': [0.5, 1, 5, 10],
        'execution_time_range': (5, 50),  # ms
        'lambda_memory': 128
    },
    'thumbnail_processing': {
        'payload_sizes_kb': [50, 100, 200, 500, 1024],
        'execution_time_range': (100, 500),  # ms
        'lambda_memory': 512
    },
    'medium_processing': {
        'payload_sizes_kb': [1024, 2048, 3072, 5120],
        'execution_time_range': (500, 2000),  # ms
        'lambda_memory': 1024
    },
    'heavy_processing': {
        'payload_sizes_kb': [3072, 5120, 8192, 10240],
        'execution_time_range': (2000, 10000),  # ms
        'lambda_memory': 2048
    }
}

TIME_WINDOWS = {
    'early_morning': {'hours': [2, 3, 4], 'description': 'Low traffic, high cold starts'},
    'morning_peak': {'hours': [8, 9, 10], 'description': 'Rising traffic'},
    'midday': {'hours': [12, 13, 14], 'description': 'Peak traffic, warm containers'},
    'evening': {'hours': [18, 19, 20], 'description': 'Second peak'},
    'late_night': {'hours': [22, 23, 0, 1], 'description': 'Declining traffic'}
}

LOAD_PATTERNS = ['low_load', 'medium_load', 'burst_load', 'ramp_load']


def get_current_time_window() -> str:
    """Determine current time window based on hour"""
    current_hour = datetime.now().hour

    for window_name, config in TIME_WINDOWS.items():
        if current_hour in config['hours']:
            return window_name

    return 'other'


def generate_workload_request(workload_type: str = None, load_pattern: str = None) -> Dict:
    """
    Generate a single workload request with realistic parameters

    Args:
        workload_type: Specific workload type or random if None
        load_pattern: Specific load pattern or random if None

    Returns:
        Dictionary with workload parameters
    """
    # Select workload type
    if workload_type is None:
        workload_type = random.choice(list(WORKLOAD_CONFIGS.keys()))

    # Select load pattern
    if load_pattern is None:
        load_pattern = random.choice(LOAD_PATTERNS)

    # Get workload config
    config = WORKLOAD_CONFIGS[workload_type]

    # Select random payload size from configured range
    payload_size_kb = random.choice(config['payload_sizes_kb'])

    # Get current time window
    time_window = get_current_time_window()

    # Generate request
    request = {
        'request_id': f"eval_{int(time.time() * 1000)}_{random.randint(1000, 9999)}",
        'workload_type': workload_type,
        'payload_size_kb': payload_size_kb,
        'time_window': time_window,
        'load_pattern': load_pattern,
        'timestamp': datetime.now().isoformat(),
        'lambda_memory_mb': config['lambda_memory']
    }

    return request


def generate_test_batch(count: int = 100, distribution: Dict = None) -> List[Dict]:
    """
    Generate a batch of test workloads with specified distribution

    Args:
        count: Number of requests to generate
        distribution: Dictionary specifying workload type distribution
                     e.g., {'lightweight_api': 0.4, 'thumbnail_processing': 0.3, ...}

    Returns:
        List of workload requests
    """
    if distribution is None:
        # Default equal distribution
        distribution = {wt: 0.25 for wt in WORKLOAD_CONFIGS.keys()}

    requests = []

    for workload_type, proportion in distribution.items():
        workload_count = int(count * proportion)

        for _ in range(workload_count):
            request = generate_workload_request(workload_type=workload_type)
            requests.append(request)

    # Shuffle to randomize order
    random.shuffle(requests)

    return requests


def save_test_batch(requests: List[Dict], filename: str):
    """Save generated requests to JSON file"""
    with open(filename, 'w') as f:
        json.dump(requests, f, indent=2)

    print(f"Saved {len(requests)} requests to {filename}")


if __name__ == '__main__':
    # Generate test batch
    print("Generating test workloads...")

    # Equal distribution test
    test_requests = generate_test_batch(count=1000)
    save_test_batch(test_requests, 'test_workloads_equal.json')

    # Realistic distribution (more lightweight, fewer heavy)
    realistic_dist = {
        'lightweight_api': 0.50,
        'thumbnail_processing': 0.30,
        'medium_processing': 0.15,
        'heavy_processing': 0.05
    }
    test_requests = generate_test_batch(count=1000, distribution=realistic_dist)
    save_test_batch(test_requests, 'test_workloads_realistic.json')

    print("\n✅ Test workload generation complete!")
    print(f"   Current time window: {get_current_time_window()}")
