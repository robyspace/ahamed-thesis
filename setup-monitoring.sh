#!/bin/bash

echo "Setting up CloudWatch monitoring and cost tracking..."

# 1. Enable Cost Explorer (if not already enabled)
aws ce get-cost-and-usage \
    --time-period Start=2025-11-10,End=2025-11-10 \
    --granularity HOURLY \
    --metrics BlendedCost \
    --group-by Type=SERVICE \
    > /dev/null 2>&1

# 2. Create CloudWatch Dashboard
cat > dashboard-config.json << 'EOF'
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          [ "AWS/Lambda", "Invocations", { "stat": "Sum" } ],
          [ ".", "Duration", { "stat": "Average" } ],
          [ ".", "Errors", { "stat": "Sum" } ],
          [ ".", "Throttles", { "stat": "Sum" } ]
        ],
        "period": 300,
        "stat": "Sum",
        "region": "eu-west-1",
        "title": "Lambda Metrics",
        "yAxis": {
          "left": {
            "min": 0
          }
        }
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          [ "AWS/ECS", "CPUUtilization", { "stat": "Average" } ],
          [ ".", "MemoryUtilization", { "stat": "Average" } ]
        ],
        "period": 300,
        "stat": "Average",
        "region": "eu-west-1",
        "title": "ECS Metrics"
      }
    },
    {
      "type": "metric",
      "properties": {
        "metrics": [
          [ "AWS/ApplicationELB", "TargetResponseTime", { "stat": "Average" } ],
          [ ".", "RequestCount", { "stat": "Sum" } ]
        ],
        "period": 300,
        "stat": "Average",
        "region": "eu-west-1",
        "title": "ALB Metrics"
      }
    }
  ]
}
EOF

aws cloudwatch put-dashboard \
    --dashboard-name HybridThesisDashboard \
    --dashboard-body file://dashboard-config.json

echo "✓ CloudWatch Dashboard created"

# 3. Create CloudWatch Alarms for cost monitoring
aws cloudwatch put-metric-alarm \
    --alarm-name hybrid-thesis-high-cost-alert \
    --alarm-description "Alert when daily costs exceed threshold" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 86400 \
    --evaluation-periods 1 \
    --threshold 50.0 \
    --comparison-operator GreaterThanThreshold

echo "✓ Cost alarm created (threshold: $50/day)"

echo ""
echo "Monitoring setup complete!"
echo "Dashboard: https://console.aws.amazon.com/cloudwatch/home?region=eu-west-1#dashboards:name=HybridThesisDashboard"