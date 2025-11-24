# Hybrid ML Router - Quick Reference

## API Endpoint
```
POST https://73jtubvdei.execute-api.eu-west-1.amazonaws.com/prod/predict
```

## Quick Test
```bash
curl -X POST https://73jtubvdei.execute-api.eu-west-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"workload_type": "lightweight_api", "payload_size_kb": 5.0, "time_window": "midday", "load_pattern": "medium_load"}'
```

## View Logs
```bash
aws logs tail /aws/lambda/hybrid-ml-router --follow
```

## Update Function
```bash
cd deployment/lambda-inference
docker build --platform linux/amd64 --provenance=false --sbom=false \
  -t 851974186570.dkr.ecr.eu-west-1.amazonaws.com/hybrid-ml-router:latest .
docker push 851974186570.dkr.ecr.eu-west-1.amazonaws.com/hybrid-ml-router:latest
aws lambda update-function-code --function-name hybrid-ml-router \
  --image-uri 851974186570.dkr.ecr.eu-west-1.amazonaws.com/hybrid-ml-router:latest
```

## CloudWatch Dashboard
https://console.aws.amazon.com/cloudwatch/home?region=eu-west-1

## Lambda Function
https://console.aws.amazon.com/lambda/home?region=eu-west-1#/functions/hybrid-ml-router

## API Gateway
https://console.aws.amazon.com/apigateway/home?region=eu-west-1#/apis/73jtubvdei

## Test Cases

### Lightweight API
```json
{"workload_type": "lightweight_api", "payload_size_kb": 5.0, "time_window": "midday", "load_pattern": "medium_load"}
```
Expected: ECS (57.6% confidence)

### Heavy Processing
```json
{"workload_type": "heavy_processing", "payload_size_kb": 5120.0, "time_window": "evening", "load_pattern": "burst_load"}
```
Expected: Lambda (100% confidence)

### Thumbnail Processing
```json
{"workload_type": "thumbnail_processing", "payload_size_kb": 200.0, "time_window": "midday", "load_pattern": "medium_load"}
```

### Medium Processing
```json
{"workload_type": "medium_processing", "payload_size_kb": 10.0, "time_window": "early_morning", "load_pattern": "low_load"}
```

## Performance
- Cold Start: ~7.8s
- Warm Invocation: ~1ms
- Cost per request: $0.00000021
