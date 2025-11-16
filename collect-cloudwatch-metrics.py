import boto3
import json
from datetime import datetime, timedelta
import pandas as pd

cloudwatch = boto3.client('cloudwatch', region_name='eu-west-1')
ce = boto3.client('ce', region_name='eu-west-1')

def get_lambda_metrics(function_name, start_time, end_time):
    """Fetch Lambda metrics from CloudWatch"""
    
    metrics = ['Invocations', 'Duration', 'Errors', 'ConcurrentExecutions']
    results = {}
    
    for metric in metrics:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/Lambda',
            MetricName=metric,
            Dimensions=[{'Name': 'FunctionName', 'Value': function_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Sum', 'Average', 'Maximum']
        )
        results[metric] = response['Datapoints']
    
    return results

def get_ecs_metrics(cluster_name, service_name, start_time, end_time):
    """Fetch ECS metrics from CloudWatch"""
    
    metrics = ['CPUUtilization', 'MemoryUtilization']
    results = {}
    
    for metric in metrics:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/ECS',
            MetricName=metric,
            Dimensions=[
                {'Name': 'ClusterName', 'Value': cluster_name},
                {'Name': 'ServiceName', 'Value': service_name}
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average', 'Maximum']
        )
        results[metric] = response['Datapoints']
    
    return results

def get_cost_data(start_date, end_date):
    """Fetch cost data from AWS Cost Explorer"""
    
    response = ce.get_cost_and_usage(
        TimePeriod={
            'Start': start_date,
            'End': end_date
        },
        Granularity='DAILY',
        Metrics=['BlendedCost', 'UsageQuantity'],
        GroupBy=[
            {'Type': 'SERVICE', 'Key': 'SERVICE'},
            {'Type': 'TAG', 'Key': 'Project'}
        ],
        Filter={
            'Tags': {
                'Key': 'Project',
                'Values': ['HybridThesis']
            }
        }
    )
    
    return response['ResultsByTime']

if __name__ == '__main__':
    # Set time range (last 24 hours)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=1)
    
    print("Collecting CloudWatch metrics...")
    
    # Lambda functions
    lambda_functions = [
        'hybrid-thesis-lightweight-api',
        'hybrid-thesis-thumbnail',
        'hybrid-thesis-medium',
        'hybrid-thesis-heavy'
    ]
    
    lambda_data = {}
    for func in lambda_functions:
        print(f"  Fetching Lambda metrics: {func}")
        lambda_data[func] = get_lambda_metrics(func, start_time, end_time)
    
    # ECS services
    ecs_services = [
        'hybrid-thesis-lightweight-service',
        'hybrid-thesis-thumbnail-service',
        'hybrid-thesis-medium-service',
        'hybrid-thesis-heavy-service'
    ]
    
    ecs_data = {}
    for service in ecs_services:
        print(f"  Fetching ECS metrics: {service}")
        ecs_data[service] = get_ecs_metrics('hybrid-thesis-cluster', service, start_time, end_time)
    
    # Cost data
    print("Fetching cost data...")
    cost_data = get_cost_data(
        start_time.strftime('%Y-%m-%d'),
        end_time.strftime('%Y-%m-%d')
    )
    
    # Save to file
    output = {
        'collection_time': datetime.utcnow().isoformat(),
        'time_range': {
            'start': start_time.isoformat(),
            'end': end_time.isoformat()
        },
        'lambda_metrics': lambda_data,
        'ecs_metrics': ecs_data,
        'cost_data': cost_data
    }
    
    filename = f"cloudwatch_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(f'data-output/{filename}', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✓ Metrics saved to: data-output/{filename}")