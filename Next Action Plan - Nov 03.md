# Next Steps Action Plan - Hybrid Serverless-Container Architecture

## Current Status

✅ **Completed**: Artillery.io dataset prepared (28,325 requests) ⚠️ **Missing**: Actual Lambda and ECS performance data for ground truth labels

---

## PHASE 1: DATA COLLECTION COMPLETION (Week 1-2)

### Option A: Full Dual-Platform Deployment (RECOMMENDED)

#### Week 1: AWS Infrastructure Setup

**1.1 Lambda Deployment**

```bash
# Create Lambda functions for each workload type
- Lightweight API (128MB memory)
- Thumbnail Processing (256MB memory)
- Medium Processing (256MB, 512MB memory)
- Heavy Processing (512MB memory)

Required Components:
- API Gateway endpoints
- CloudWatch logging with custom metrics
- Cost tracking via AWS Cost Explorer
- X-Ray tracing for detailed performance analysis
```

**1.2 ECS Deployment**

```bash
# Deploy equivalent containerized applications
- ECS Fargate cluster with matching resource allocations
- Application Load Balancer
- CloudWatch Container Insights
- Equivalent logging and monitoring

Resource Matching:
- 128MB Lambda → 0.25 vCPU, 512MB memory ECS
- 256MB Lambda → 0.5 vCPU, 1GB memory ECS
- 512MB Lambda → 1 vCPU, 2GB memory ECS
```

**1.3 Dual-Target Artillery Configuration**

```yaml
# Modified Artillery config to hit both platforms
config:
  target: "https://api-gateway-url.amazonaws.com"  # Lambda
  phases:
    - duration: 3600
      arrivalRate: 10
      
scenarios:
  - name: "Dual Platform Test"
    flow:
      - post:
          url: "/lambda/validate"
          headers:
            X-Target-Platform: "Lambda"
          json:
            payload: "{{ payload }}"
          capture:
            - json: "$.latency"
              as: "lambda_latency"
            - json: "$.cost"
              as: "lambda_cost"
      
      - post:
          url: "https://alb-url.amazonaws.com/validate"  # ECS
          headers:
            X-Target-Platform: "ECS"
          json:
            payload: "{{ payload }}"
          capture:
            - json: "$.latency"
              as: "ecs_latency"
            - json: "$.cost"
              as: "ecs_cost"
```

#### Week 2: Data Collection Campaign

**2.1 Run Continuous Tests**

```bash
# Run for 7-10 days to collect 50,000+ dual-platform measurements
artillery run dual-platform-workload.yml --output results-$(date +%Y%m%d).json

# Schedule via cron for 24/7 collection
0 * * * * artillery run microservice-api.yml --output lambda-api-$(date +\%Y\%m\%d-\%H).json
0 * * * * artillery run microservice-api.yml --target https://ecs-alb.amazonaws.com --output ecs-api-$(date +\%Y\%m\%d-\%H).json
```

**2.2 Metrics to Capture Per Request**

```json
{
  "request_id": "req_12345",
  "timestamp": "2025-11-03T10:00:00Z",
  "workload_type": "lightweight_api",
  "payload_size_kb": 0.05,
  "endpoint": "/api/validate",
  
  "lambda_metrics": {
    "response_time_ms": 180,
    "cold_start": false,
    "memory_used_mb": 95,
    "billed_duration_ms": 200,
    "cost_usd": 0.0000004167,
    "status_code": 200
  },
  
  "ecs_metrics": {
    "response_time_ms": 245,
    "memory_used_mb": 120,
    "cpu_utilization": 15.3,
    "cost_usd": 0.0000012500,
    "status_code": 200
  },
  
  "system_state": {
    "concurrent_requests": 23,
    "hour_of_day": 10,
    "day_of_week": 1
  }
}
```

### Option B: Simulation-Based Approach (FALLBACK - Use only if AWS deployment impossible)

If you cannot deploy on AWS due to cost/time constraints:

**B.1 Use Your Current Dataset as Base**

- Your existing 28,325 requests provide realistic workload characteristics

**B.2 Generate Synthetic Performance Labels Using Literature-Based Models**

```python
# Based on research papers and AWS documentation
def estimate_lambda_performance(row):
    """Estimate Lambda performance based on research benchmarks"""
    base_latency = row['estimated_exec_time_ms']
    
    # Cold start penalty (10-15% of requests, +100-800ms)
    cold_start_prob = 0.12
    cold_start = np.random.binomial(1, cold_start_prob)
    cold_start_penalty = np.random.uniform(100, 800) if cold_start else 0
    
    # Lambda overhead (~5-15ms)
    lambda_overhead = np.random.uniform(5, 15)
    
    # Cost calculation (AWS pricing)
    memory_gb_seconds = (row['memory_requirement_mb'] / 1024) * (base_latency / 1000)
    cost = memory_gb_seconds * 0.0000166667  # $0.0000166667 per GB-second
    
    return {
        'lambda_latency_ms': base_latency + lambda_overhead + cold_start_penalty,
        'lambda_cost_usd': cost + 0.0000002,  # +$0.20 per 1M requests
        'lambda_cold_start': cold_start
    }

def estimate_ecs_performance(row):
    """Estimate ECS performance based on container benchmarks"""
    base_latency = row['estimated_exec_time_ms']
    
    # Container has lower overhead but always-on cost
    container_overhead = np.random.uniform(2, 8)
    
    # ECS Fargate pricing (continuous billing)
    vcpu_cost_per_hour = 0.04048  # per vCPU
    memory_cost_per_hour = 0.004445  # per GB
    
    # Calculate for request duration
    duration_hours = base_latency / 3600000
    vcpu_allocation = row['memory_requirement_mb'] / 512  # rough estimate
    memory_gb = row['memory_requirement_mb'] / 1024
    
    cost = (vcpu_cost_per_hour * vcpu_allocation + memory_cost_per_hour * memory_gb) * duration_hours
    
    return {
        'ecs_latency_ms': base_latency + container_overhead,
        'ecs_cost_usd': cost,
        'ecs_memory_mb': row['memory_requirement_mb'] * 1.2  # containers use more
    }
```

**B.3 Validation Against Azure Dataset**

- Compare your synthetic distributions with Azure Functions traces
- Adjust parameters to match real-world patterns

---

## PHASE 2: DATASET PREPARATION & LABELING (Week 3)

### Step 2.1: Merge Lambda and ECS Data

```python
import pandas as pd
import numpy as np

# Load your consolidated dataset
df = pd.read_csv('consolidated_workload_dataset.csv')

# If using Option A (real data)
lambda_data = pd.read_json('lambda_metrics_combined.json')
ecs_data = pd.read_json('ecs_metrics_combined.json')

merged_df = pd.merge(
    lambda_data,
    ecs_data,
    on=['request_id', 'timestamp'],
    suffixes=('_lambda', '_ecs')
)

# If using Option B (simulated data)
# Apply estimation functions
lambda_perf = df.apply(estimate_lambda_performance, axis=1, result_type='expand')
ecs_perf = df.apply(estimate_ecs_performance, axis=1, result_type='expand')

merged_df = pd.concat([df, lambda_perf, ecs_perf], axis=1)
```

### Step 2.2: Create Ground Truth Labels

```python
# Label assignment logic from your plan
def assign_optimal_platform(row):
    """
    Determine optimal platform based on performance and cost
    
    Label = 1 (Lambda) if:
    - Lambda is faster AND
    - Lambda cost is within 10% of ECS cost
    
    Otherwise Label = 0 (ECS)
    """
    lambda_faster = row['lambda_latency_ms'] < row['ecs_latency_ms']
    cost_acceptable = row['lambda_cost_usd'] <= (row['ecs_cost_usd'] * 1.1)
    
    return 1 if (lambda_faster and cost_acceptable) else 0

merged_df['optimal_platform'] = merged_df.apply(assign_optimal_platform, axis=1)

# Feature engineering
merged_df['latency_ratio'] = merged_df['lambda_latency_ms'] / merged_df['ecs_latency_ms']
merged_df['cost_ratio'] = merged_df['lambda_cost_usd'] / merged_df['ecs_cost_usd']
merged_df['cost_performance_score'] = (
    merged_df['latency_ratio'] * 0.6 + 
    merged_df['cost_ratio'] * 0.4
)
```

### Step 2.3: Dataset Statistics & Validation

```python
print("Label Distribution:")
print(merged_df['optimal_platform'].value_counts())
print("\nLabel Distribution by Workload Type:")
print(pd.crosstab(merged_df['workload_category'], merged_df['optimal_platform'], 
                   normalize='index'))

# Ensure balanced classes (if needed)
from imblearn.over_sampling import SMOTE

if merged_df['optimal_platform'].value_counts().min() < len(merged_df) * 0.3:
    print("\nWarning: Imbalanced classes detected. Consider SMOTE oversampling.")
```

---

## PHASE 3: FEATURE ENGINEERING (Week 4)

### Step 3.1: Feature Selection

```python
# Select features for ML model
features = [
    # Request characteristics
    'payload_size_kb',
    'payload_size_bytes',
    'estimated_exec_time_ms',
    'memory_requirement_mb',
    
    # Temporal features
    'hour_of_day',
    'day_of_week',
    
    # System state
    'concurrent_requests',  # Need to calculate this
    
    # Workload type (one-hot encoded)
    'workload_category',
    
    # Historical patterns
    'recent_request_rate',  # Calculate from timestamp windows
]

# One-hot encode categorical variables
df_encoded = pd.get_dummies(merged_df, columns=['workload_category'], prefix='workload')

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numeric_features = ['payload_size_kb', 'estimated_exec_time_ms', 'memory_requirement_mb']
df_encoded[numeric_features] = scaler.fit_transform(df_encoded[numeric_features])
```

### Step 3.2: Feature Importance Analysis (Preliminary)

```python
from sklearn.ensemble import RandomForestClassifier

# Quick feature importance check
X = df_encoded[features]
y = df_encoded['optimal_platform']

rf_temp = RandomForestClassifier(n_estimators=50, random_state=42)
rf_temp.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
```

---

## PHASE 4: ML MODEL DEVELOPMENT (Weeks 5-6)

### Step 4.1: Train-Test Split

```python
from sklearn.model_selection import train_test_split

X = df_encoded[features]
y = df_encoded['optimal_platform']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
```

### Step 4.2: Model Training & Evaluation

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb

# 1. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_score = rf_model.score(X_test, y_test)

print("Random Forest Results:")
print(f"Accuracy: {rf_score:.4f}")
print(classification_report(y_test, rf_pred))

# 2. XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_score = xgb_model.score(X_test, y_test)

print("\nXGBoost Results:")
print(f"Accuracy: {xgb_score:.4f}")
print(classification_report(y_test, xgb_pred))

# 3. Neural Network
nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42
)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
nn_score = nn_model.score(X_test, y_test)

print("\nNeural Network Results:")
print(f"Accuracy: {nn_score:.4f}")
print(classification_report(y_test, nn_pred))

# Select best model
best_model = max([
    ('Random Forest', rf_model, rf_score),
    ('XGBoost', xgb_model, xgb_score),
    ('Neural Network', nn_model, nn_score)
], key=lambda x: x[2])

print(f"\nBest Model: {best_model[0]} with accuracy {best_model[2]:.4f}")
```

### Step 4.3: Model Optimization

```python
from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning for best model
if best_model[0] == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10, 15]
    }
elif best_model[0] == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 10, 15],
        'learning_rate': [0.01, 0.1, 0.2]
    }

grid_search = GridSearchCV(
    best_model[1],
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
optimized_model = grid_search.best_estimator_

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")
```

---

## PHASE 5: ROUTER IMPLEMENTATION (Weeks 7-8)

### Step 5.1: Save Trained Model

```python
import joblib

# Save the best model
joblib.dump(optimized_model, 'intelligent_router_model.pkl')
joblib.dump(scaler, 'feature_scaler.pkl')

# Save feature names for inference
import json
with open('model_features.json', 'w') as f:
    json.dump(features, f)
```

### Step 5.2: Create Router Service

```python
# intelligent_router.py
import joblib
import numpy as np
import json

class IntelligentRouter:
    def __init__(self):
        self.model = joblib.load('intelligent_router_model.pkl')
        self.scaler = joblib.load('feature_scaler.pkl')
        
        with open('model_features.json', 'r') as f:
            self.features = json.load(f)
    
    def extract_features(self, request):
        """Extract features from incoming request"""
        return {
            'payload_size_kb': len(json.dumps(request.payload)) / 1024,
            'estimated_exec_time_ms': self.estimate_execution_time(request),
            'memory_requirement_mb': self.estimate_memory(request),
            'hour_of_day': datetime.now().hour,
            'day_of_week': datetime.now().weekday(),
            'concurrent_requests': self.get_current_load(),
            # ... other features
        }
    
    def route_request(self, request):
        """
        Determine optimal platform for request
        Returns: 'lambda' or 'ecs'
        """
        features = self.extract_features(request)
        feature_vector = np.array([features[f] for f in self.features]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict optimal platform
        prediction = self.model.predict(feature_vector_scaled)[0]
        
        return 'lambda' if prediction == 1 else 'ecs'
```

### Step 5.3: Deploy Router as Lambda Function

```python
# lambda_router.py
import json
import boto3
from intelligent_router import IntelligentRouter

router = IntelligentRouter()

def lambda_handler(event, context):
    """
    AWS Lambda function for intelligent routing
    """
    # Parse incoming request
    request_body = json.loads(event['body'])
    
    # Determine optimal platform
    target_platform = router.route_request(request_body)
    
    # Route to appropriate backend
    if target_platform == 'lambda':
        response = invoke_lambda_backend(request_body)
    else:
        response = invoke_ecs_backend(request_body)
    
    return {
        'statusCode': 200,
        'headers': {
            'X-Routed-To': target_platform
        },
        'body': json.dumps(response)
    }

def invoke_lambda_backend(request):
    lambda_client = boto3.client('lambda')
    response = lambda_client.invoke(
        FunctionName='worker-lambda',
        Payload=json.dumps(request)
    )
    return json.loads(response['Payload'].read())

def invoke_ecs_backend(request):
    # HTTP request to ECS service
    import requests
    response = requests.post(
        'https://ecs-service.amazonaws.com/process',
        json=request
    )
    return response.json()
```

---

## PHASE 6: EVALUATION & TESTING (Weeks 9-10)

### Step 6.1: Comparative Testing

Test three configurations:

1. **Lambda-Only**: All requests to Lambda
2. **ECS-Only**: All requests to ECS
3. **ML-Hybrid**: Intelligent routing

```python
# test_comparison.py
import pandas as pd
import numpy as np

def run_comparison_test(test_dataset, duration_minutes=60):
    """
    Run comparative testing on three configurations
    """
    results = {
        'lambda_only': [],
        'ecs_only': [],
        'ml_hybrid': []
    }
    
    for request in test_dataset:
        # Lambda-only
        lambda_metrics = send_to_lambda(request)
        results['lambda_only'].append(lambda_metrics)
        
        # ECS-only
        ecs_metrics = send_to_ecs(request)
        results['ecs_only'].append(ecs_metrics)
        
        # ML-hybrid
        routed_platform = router.route_request(request)
        if routed_platform == 'lambda':
            hybrid_metrics = send_to_lambda(request)
        else:
            hybrid_metrics = send_to_ecs(request)
        hybrid_metrics['routed_to'] = routed_platform
        results['ml_hybrid'].append(hybrid_metrics)
    
    return results

# Run tests
test_results = run_comparison_test(test_dataset, duration_minutes=60)

# Calculate metrics
for config in ['lambda_only', 'ecs_only', 'ml_hybrid']:
    df = pd.DataFrame(test_results[config])
    
    print(f"\n{config.upper()} Results:")
    print(f"Average Latency: {df['latency_ms'].mean():.2f}ms")
    print(f"P95 Latency: {df['latency_ms'].quantile(0.95):.2f}ms")
    print(f"P99 Latency: {df['latency_ms'].quantile(0.99):.2f}ms")
    print(f"Total Cost: ${df['cost_usd'].sum():.4f}")
    print(f"Error Rate: {(df['status_code'] != 200).mean() * 100:.2f}%")
```

---

## DELIVERABLES CHECKLIST

### For Your Professor/Thesis:

- [ ] **Dataset Documentation**
    
    - Source (Artillery.io + AWS/simulated)
    - Size (50,000+ requests minimum)
    - Feature descriptions
    - Label distribution
- [ ] **Methodology Chapter**
    
    - Data collection process
    - Labeling logic with justification
    - Feature engineering rationale
- [ ] **ML Model Results**
    
    - Comparison of RF, XGBoost, Neural Network
    - Accuracy, Precision, Recall, F1-score
    - Feature importance analysis
    - Confusion matrices
- [ ] **Implementation**
    
    - Router code
    - Deployment architecture
    - API documentation
- [ ] **Evaluation Results**
    
    - Comparative performance (Lambda vs ECS vs Hybrid)
    - Cost analysis
    - Latency distributions
    - Real-world case study
- [ ] **Visualizations**
    
    - Feature importance plots
    - Performance comparison charts
    - Cost-latency trade-off graphs
    - Confusion matrices

---

## TIMELINE SUMMARY

|Week|Phase|Key Activities|
|---|---|---|
|1-2|Data Collection|AWS deployment OR simulation generation|
|3|Dataset Prep|Merging, labeling, feature engineering|
|4|EDA & Features|Feature selection, importance analysis|
|5-6|ML Training|Train 3 models, optimize best one|
|7-8|Router Implementation|Build & deploy intelligent router|
|9-10|Evaluation|Run comparative tests, analyze results|
|11-12|Documentation|Write thesis chapters, create visualizations|

---

## IMMEDIATE ACTION ITEMS (This Week)

1. **DECIDE**: Option A (real AWS deployment) vs Option B (simulation)
    
    - Option A is strongly preferred but requires AWS resources
    - Option B is acceptable but must be properly justified
2. **If Option A**:
    
    - Set up AWS account with appropriate permissions
    - Deploy sample Lambda function
    - Deploy sample ECS container
    - Test Artillery dual-targeting
3. **If Option B**:
    
    - Implement performance estimation functions
    - Validate against Azure dataset statistics
    - Document assumptions clearly
4. **Schedule meeting with professor** to confirm approach
    
5. **Start Phase 2** (Dataset Preparation) once data is ready
    

---

## QUESTIONS TO RESOLVE

1. Do you have access to AWS for real deployment?
2. What is your AWS budget/resource limit?
3. Is simulation-based approach acceptable to your professor?
4. Which ML frameworks are you most comfortable with?
5. Do you need help with specific implementation steps?

Let me know which option you want to pursue, and I can provide more detailed code and configurations!