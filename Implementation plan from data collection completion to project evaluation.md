## **PHASE 1: DATASET PREPARATION & FEATURE ENGINEERING (2-3 days)**

**Data Consolidation**

- Merge Lambda and ECS metrics by `request_id` and `timestamp`
- Verify data completeness: minimum 50,000 requests with diverse workload distribution
- Remove duplicates and handle missing values
- Validate data quality: check for errors, outliers, and anomalies

**Ground Truth Labeling**

- Create `optimal_platform` binary label (0=ECS, 1=Lambda)
- Labeling logic: Lambda optimal if `(lambda_latency < ecs_latency) AND (lambda_cost ≤ ecs_cost * 1.1)`
- Document class distribution and address potential imbalance

**Feature Engineering**

- Core features: payload_size_kb, execution_time_ms, memory_requirement_mb, workload_type
- Temporal features: time_of_day, day_of_week, concurrent_request_count
- System load indicators: burst_detection (>10 req/sec), cold_start_probability
- Derived features: execution_ratio, cost_ratio, resource_utilization
- Encode categorical variables: one-hot encoding for workload categories
- Feature scaling: StandardScaler for numeric features

**Exploratory Data Analysis**

- Analyze feature distributions and correlations
- Visualize Lambda vs ECS performance patterns by workload type
- Identify preliminary feature importance patterns

---

## **PHASE 2: ML MODEL DEVELOPMENT & TRAINING (4-5 days)**

**Train-Test Split**

- 80-20 stratified split to maintain class distribution
- Set random_state=42 for reproducibility
- Verify sufficient samples in both classes

**Model Training (3 Classifiers)**

**Random Forest** (baseline, interpretable)

- n_estimators=200, max_depth=15, min_samples_split=10
- class_weight='balanced' for imbalance handling
- Extract feature importance rankings

**XGBoost** (best accuracy target)

- n_estimators=200, max_depth=10, learning_rate=0.1
- Scale_pos_weight for class imbalance
- Early stopping with validation set

**Neural Network** (comparison only if needed)

- MLP with 2 hidden layers (100, 50 neurons)
- ReLU activation, Adam optimizer
- Consider if accuracy target not met with tree-based models

**Model Evaluation Metrics**

- Accuracy, Precision, Recall, F1-score per class
- Confusion matrix analysis
- ROC-AUC curves
- Per-workload-type performance breakdown
- Target: 85%+ overall accuracy

**Handling Class Imbalance**

- Apply SMOTE oversampling if recall < 60%
- Use class weights in all models
- Validate balanced performance across both classes

**Hyperparameter Optimization**

- Grid search or random search on best-performing model
- Cross-validation (5-fold stratified)
- Document optimal parameter combinations

**Feature Importance Analysis**

- Rank features by importance scores
- Visualize top 10 most influential features
- Validate feature selection aligns with domain knowledge

**Model Selection & Saving**

- Select best model based on accuracy + F1-score
- Save using joblib or pickle for deployment
- Document model versioning

---

## **PHASE 3: INTELLIGENT ROUTER IMPLEMENTATION (3-4 days)**

**Core Router Components**

**Feature Extraction Module**

- Parse incoming request attributes in real-time
- Extract: payload size, request type, current system load, time features
- Handle missing or invalid feature values gracefully

**Model Inference Engine**

- Load trained ML model (joblib)
- Implement <5ms inference latency requirement
- Add confidence threshold logic (e.g., route to Lambda if confidence >0.7)

**Platform Routing Logic**

- Route to AWS Lambda if model predicts 1
- Route to AWS ECS if model predicts 0
- Implement fallback strategy for edge cases

**Implementation Options**

**Option A: AWS Lambda Router** (recommended for simplicity)

- Python Lambda function with scikit-learn layer
- API Gateway trigger
- Fast cold start with pre-loaded model

**Option B: ECS Container Router** (production-ready)

- Containerized Flask/FastAPI service
- Deploy on existing ECS cluster
- Better for complex routing logic

**Router API Design**

- RESTful endpoint: POST /route
- Request payload: {workload_type, payload_size, timestamp}
- Response: {target_platform: "lambda"|"ecs", confidence: 0.XX}

**Logging & Monitoring**

- Log all routing decisions to CloudWatch
- Track routing accuracy in production
- Monitor latency overhead from router

---

## **PHASE 4: COMPARATIVE EVALUATION (4-5 days)**

**Three Configuration Testing**

**1. Lambda-Only Baseline**

- Route all workload types to Lambda
- Collect: latency (mean, P50, P95, P99), total cost, error rate

**2. ECS-Only Baseline**

- Route all workload types to ECS
- Collect same metrics as Lambda-only

**3. ML-Hybrid (Your Solution)**

- Use intelligent router for dynamic routing
- Track routing decisions by workload type
- Measure routing overhead (<5ms target)

**Evaluation Test Design**

- Run identical workload mix across all three configurations
- Duration: 2-3 hours per configuration with 10,000+ requests
- Use Artillery.io with same request patterns as training data

**Performance Metrics Collection**

- **Latency**: mean, median, P95, P99 response times by configuration
- **Cost**: total cost, cost per request, cost savings vs baselines
- **Throughput**: requests per second, concurrent request handling
- **Accuracy**: routing decision correctness (compare actual vs predicted optimal)
- **Error rates**: 4xx/5xx errors across platforms

**Comparative Analysis**

- Generate side-by-side performance comparison tables
- Calculate percentage improvements: cost savings, latency reduction
- Analyze routing patterns: % routed to Lambda vs ECS by workload
- Identify where ML-hybrid excels vs where baselines perform better

**Visualization Creation**

- Latency distribution histograms (all 3 configurations)
- Cost-latency scatter plots
- Feature importance bar charts
- Confusion matrix heatmaps
- Routing decision distribution pie charts
- Performance improvement percentage bar charts

---

## **PHASE 5: DOCUMENTATION & THESIS INTEGRATION (Ongoing)**

**Results Chapter**

- Present evaluation metrics in tables and charts
- Discuss cost-performance trade-offs achieved
- Analyze routing accuracy and patterns
- Highlight cost savings percentages and latency improvements

**Implementation Chapter**

- Document router architecture with diagrams
- Explain ML model selection rationale
- Describe deployment approach

**Discussion Section**

- Limitations: dataset size, AWS region specificity, workload types tested
- Threats to validity: cold start variability, cost estimation accuracy
- Future work: multi-cloud support, online learning, additional workload types

**Key Deliverables Checklist**

- ✓ Labeled dataset with 50K+ requests
- ✓ Trained ML models with 85%+ accuracy
- ✓ Deployed intelligent router
- ✓ Comparative evaluation results (3 configurations)
- ✓ Performance visualizations (6+ charts)
- ✓ Cost analysis with percentage savings
- ✓ Complete methodology documentation

---

**CRITICAL SUCCESS FACTORS**

- Achieve 85%+ ML model accuracy before router deployment
- Ensure router adds <5ms latency overhead
- Demonstrate measurable cost savings in hybrid configuration
- Document all assumptions and limitations clearly
- Maintain reproducibility with versioned code and datasets

**Estimated Timeline: 14-17 days total** from data collection completion to evaluation results ready for thesis integration.