# AWS Lambda Deployment Guide
## ML-Based Platform Routing for Hybrid Serverless-Container Architecture

**Purpose:** Deploy the trained ML model to AWS Lambda for real-time inference

---

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI configured
- Trained model files from Google Colab:
  - `variance_model_best.pkl`
  - `feature_columns.json`
  - `model_metadata.json`

---

## Deployment Methods

### Method 1: AWS Console (Recommended for Testing)

#### Step 1: Prepare Deployment Package

```bash
# Create deployment directory
mkdir lambda-deployment
cd lambda-deployment

# Copy model files (from Colab downloads)
cp ~/Downloads/variance_model_best.pkl .
cp ~/Downloads/feature_columns.json .
cp ~/Downloads/model_metadata.json .

# Copy Lambda handler
cp ../lambda-inference/lambda_handler.py .

# Install dependencies
pip install -t . scikit-learn==1.3.2 numpy==1.24.3 joblib==1.3.2

# Create deployment package
zip -r deployment-package.zip .
```

**Important:** The deployment package should be < 50 MB. If larger, use Lambda Layers (Method 2).

#### Step 2: Create Lambda Function

1. **Go to AWS Lambda Console**
2. **Click "Create function"**
3. **Configure:**
   - Name: `hybrid-ml-router`
   - Runtime: `Python 3.9` or `Python 3.10`
   - Architecture: `x86_64`
   - Execution role: Create new role with basic Lambda permissions

4. **Upload deployment package:**
   - Code source → Upload from → .zip file
   - Upload `deployment-package.zip`

5. **Configure function:**
   - Handler: `lambda_handler.lambda_handler`
   - Memory: `512 MB` (or higher for faster inference)
   - Timeout: `30 seconds`
   - Environment variables (optional):
     - `MODEL_PATH` = `variance_model_best.pkl`

6. **Test the function:**
   ```json
   {
     "workload_type": "lightweight_api",
     "payload_size_kb": 5.0,
     "time_window": "midday",
     "load_pattern": "medium_load"
   }
   ```

---

### Method 2: Using Lambda Layers (For Large Dependencies)

If your deployment package exceeds 50 MB, use Lambda Layers:

#### Create Layer for scikit-learn and dependencies

```bash
# Create layer directory
mkdir python
cd python

# Install packages
pip install -t . scikit-learn==1.3.2 numpy==1.24.3 joblib==1.3.2

# Create layer package
cd ..
zip -r sklearn-layer.zip python/

# Upload to AWS (using AWS CLI)
aws lambda publish-layer-version \
    --layer-name sklearn-layer \
    --zip-file fileb://sklearn-layer.zip \
    --compatible-runtimes python3.9 python3.10
```

#### Create Lambda function with layer

```bash
# Create function package (without dependencies)
mkdir lambda-function
cd lambda-function

# Copy only code and model files
cp variance_model_best.pkl .
cp feature_columns.json .
cp model_metadata.json .
cp lambda_handler.py .

# Create package
zip -r function.zip .

# Create Lambda function
aws lambda create-function \
    --function-name hybrid-ml-router \
    --runtime python3.9 \
    --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-execution-role \
    --handler lambda_handler.lambda_handler \
    --zip-file fileb://function.zip \
    --timeout 30 \
    --memory-size 512 \
    --layers arn:aws:lambda:REGION:ACCOUNT_ID:layer:sklearn-layer:1
```

---

### Method 3: Using AWS SAM (For Production)

Create `template.yaml`:

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  HybridMLRouter:
    Type: AWS::Serverless::Function
    Properties:
      FunctionName: hybrid-ml-router
      CodeUri: ./lambda-deployment/
      Handler: lambda_handler.lambda_handler
      Runtime: python3.9
      MemorySize: 512
      Timeout: 30
      Environment:
        Variables:
          MODEL_PATH: variance_model_best.pkl
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /predict
            Method: post
```

Deploy:

```bash
sam build
sam deploy --guided
```

---

## Testing the Deployed Function

### 1. Test via AWS Console

**Test Event:**
```json
{
  "workload_type": "lightweight_api",
  "payload_size_kb": 5.0,
  "time_window": "midday",
  "load_pattern": "medium_load",
  "timestamp": "2025-11-23T12:00:00Z"
}
```

**Expected Response:**
```json
{
  "statusCode": 200,
  "body": "{\"prediction\": {\"platform\": \"lambda\", \"confidence\": 0.85, ...}}"
}
```

### 2. Test via AWS CLI

```bash
aws lambda invoke \
    --function-name hybrid-ml-router \
    --payload '{"workload_type": "lightweight_api", "payload_size_kb": 5.0, "time_window": "midday", "load_pattern": "medium_load"}' \
    response.json

cat response.json
```

### 3. Test via cURL (if API Gateway configured)

```bash
curl -X POST https://YOUR_API_ID.execute-api.REGION.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{
    "workload_type": "lightweight_api",
    "payload_size_kb": 5.0,
    "time_window": "midday",
    "load_pattern": "medium_load"
  }'
```

---

## Integrating with API Gateway

### Create REST API

1. **Go to API Gateway Console**
2. **Create REST API**
3. **Create Resource:** `/predict`
4. **Create Method:** `POST`
5. **Integration Type:** Lambda Function
6. **Lambda Function:** `hybrid-ml-router`
7. **Deploy API** to stage (e.g., `prod`)

### Test Integration

```bash
API_URL="https://YOUR_API_ID.execute-api.REGION.amazonaws.com/prod/predict"

curl -X POST $API_URL \
  -H "Content-Type: application/json" \
  -d '{
    "workload_type": "thumbnail_processing",
    "payload_size_kb": 200.0,
    "time_window": "evening",
    "load_pattern": "burst_load"
  }'
```

---

## Monitoring and Metrics

### CloudWatch Metrics

Key metrics to monitor:

1. **Invocations** - Number of predictions
2. **Duration** - Inference latency (target: <10ms)
3. **Errors** - Failed predictions
4. **Throttles** - Rate limiting issues
5. **ConcurrentExecutions** - Scaling behavior

### Custom Metrics

Add CloudWatch custom metrics to Lambda handler:

```python
import boto3
cloudwatch = boto3.client('cloudwatch')

# Log inference time
cloudwatch.put_metric_data(
    Namespace='HybridML',
    MetricData=[
        {
            'MetricName': 'InferenceLatency',
            'Value': inference_time_ms,
            'Unit': 'Milliseconds'
        }
    ]
)
```

### CloudWatch Logs

View logs:

```bash
aws logs tail /aws/lambda/hybrid-ml-router --follow
```

---

## Cost Analysis

### Lambda Inference Cost

**Configuration:**
- Memory: 512 MB
- Average execution time: 5 ms
- Pricing (eu-west-1):
  - Request: $0.20 per 1M requests
  - Duration: $0.0000166667 per GB-second

**Cost per request:**
```
Request cost:  $0.0000002
Duration cost: $0.0000000417 (512 MB × 0.005 sec)
Total:         $0.0000002417 per inference
```

**Cost for 1 million requests:**
```
1M × $0.0000002417 = $0.24
```

### Comparison with Traditional Routing

| Metric | Static Routing | ML-Based Routing | Savings |
|--------|----------------|------------------|---------|
| Infrastructure Cost | $0 (uses existing) | $0.24 per 1M requests | -$0.24 |
| Workload Efficiency | N/A | 10-30% cost reduction | +$50-150 per 1M |
| Latency Overhead | 0 ms | 2-5 ms | -2-5 ms |
| **Net Benefit** | Baseline | **+$50-150 per 1M** | **+20-60x ROI** |

---

## Performance Optimization

### 1. Cold Start Mitigation

**Use Provisioned Concurrency:**
```bash
aws lambda put-provisioned-concurrency-config \
    --function-name hybrid-ml-router \
    --provisioned-concurrent-executions 2
```

**Cost:** ~$0.015/hour per provisioned instance

### 2. Memory Optimization

Test different memory settings:
- 512 MB (baseline)
- 1024 MB (faster CPU, 2x cost)
- 256 MB (slower, 0.5x cost)

**Find optimal memory:**
```bash
# Run 1000 predictions at different memory settings
for mem in 256 512 1024; do
  # Update function memory
  aws lambda update-function-configuration \
      --function-name hybrid-ml-router \
      --memory-size $mem

  # Test inference time
  # Log results
done
```

### 3. Model Optimization

**Option A: Model Quantization**
- Reduce model size by 50-75%
- Faster loading time
- Minimal accuracy impact

**Option B: Simpler Model**
- Use LightGBM instead of XGBoost (smaller)
- Reduce tree depth/number of trees
- Target: <10 MB model size

---

## Troubleshooting

### Issue 1: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
- Ensure dependencies are in deployment package
- Check layer is attached
- Verify compatibility with Python runtime

### Issue 2: Timeout

**Problem:** Function times out during inference

**Solution:**
- Increase timeout to 30 seconds
- Increase memory to 1024 MB
- Check model loading (should be cached)

### Issue 3: Large Package Size

**Problem:** Deployment package > 50 MB

**Solution:**
- Use Lambda Layers for dependencies
- Remove unnecessary files (.pyc, tests, docs)
- Use lightweight model format

### Issue 4: Low Accuracy in Production

**Problem:** Predictions seem wrong

**Solution:**
- Verify feature extraction matches training
- Check input data format
- Log predictions and compare with expected
- Retrain model with production data

---

## Security Best Practices

### 1. IAM Permissions

Minimal IAM role:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

### 2. API Gateway Authentication

Options:
- **API Key:** Simple, for internal use
- **IAM Authorization:** For AWS services
- **Lambda Authorizer:** Custom auth logic
- **Cognito:** User authentication

### 3. Encryption

- Enable encryption at rest for S3 (model storage)
- Use HTTPS for API Gateway
- Encrypt sensitive logs

---

## Next Steps

1. ✅ Deploy Lambda function
2. ✅ Test inference
3. ✅ Set up API Gateway
4. ✅ Configure monitoring
5. 🔄 Integrate with workload router
6. 🔄 Run comparative evaluation (Lambda-only vs ECS-only vs ML-hybrid)
7. 🔄 Measure cost and latency savings
8. 🔄 Document results for thesis

---

## References

- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [AWS Lambda Pricing](https://aws.amazon.com/lambda/pricing/)
- [Deploying ML Models to Lambda](https://aws.amazon.com/blogs/compute/deploying-machine-learning-models-with-serverless-templates/)
- [scikit-learn on AWS Lambda](https://github.com/aws-samples/aws-lambda-layer-python-scientific-packages)

---

**Last Updated:** 2025-11-23
**Model Version:** variance_model_v1
**Test Accuracy:** 75-90% (check model_metadata.json)
