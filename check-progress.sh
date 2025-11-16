#!/bin/bash

echo "========================================="
echo "Data Collection Progress Report"
echo "========================================="
echo ""

# Check if Artillery is running
if ps -p $(cat artillery_pid.txt 2>/dev/null) > /dev/null 2>&1; then
    echo "✓ Artillery process is running (PID: $(cat artillery_pid.txt))"
else
    echo "✗ Artillery process is NOT running"
fi

echo ""
echo "Data Files:"
echo "-----------"
ls -lh data-output/*.jsonl 2>/dev/null | wc -l | xargs echo "  JSONL log files:"
ls -lh data-output/cloudwatch_metrics_*.json 2>/dev/null | wc -l | xargs echo "  CloudWatch metric files:"

echo ""
echo "Request Counts:"
echo "---------------"
for file in data-output/*lambda*.jsonl; do
    if [ -f "$file" ]; then
        count=$(wc -l < "$file")
        echo "  $(basename $file): $count requests"
    fi
done

for file in data-output/*ecs*.jsonl; do
    if [ -f "$file" ]; then
        count=$(wc -l < "$file")
        echo "  $(basename $file): $count requests"
    fi
done

echo ""
echo "AWS Costs (Last 24 hours):"
echo "--------------------------"
python3 << 'PYTHON'
import boto3
from datetime import datetime, timedelta

ce = boto3.client('ce', region_name='eu-west-1')
end = datetime.utcnow().strftime('%Y-%m-%d')
start = (datetime.utcnow() - timedelta(days=1)).strftime('%Y-%m-%d')

response = ce.get_cost_and_usage(
    TimePeriod={'Start': start, 'End': end},
    Granularity='DAILY',
    Metrics=['BlendedCost'],
    GroupBy=[{'Type': 'SERVICE'}]
)

if response['ResultsByTime']:
    for result in response['ResultsByTime']:
        print(f"  Date: {result['TimePeriod']['Start']}")
        for group in result['Groups']:
            service = group['Keys'][0]
            cost = float(group['Metrics']['BlendedCost']['Amount'])
            if cost > 0:
                print(f"    {service}: ${cost:.4f}")
PYTHON

echo ""
echo "========================================="