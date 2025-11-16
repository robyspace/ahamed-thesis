#!/bin/bash

export AWS_REGION="eu-west-1"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_URI=$(aws ecr describe-repositories --repository-names hybrid-thesis-app --query 'repositories[0].repositoryUri' --output text)

echo "Setting up ECS infrastructure..."
echo ""

# 0. Create ECS service-linked role first (one-time per AWS account)
echo "Step 0: Ensuring ECS service-linked role exists..."
aws iam create-service-linked-role --aws-service-name ecs.amazonaws.com 2>/dev/null || echo "  (Service-linked role already exists - this is fine)"
echo "✓ ECS service-linked role ready"
echo ""

# Wait a moment for the role to propagate
sleep 5

# 1. Create ECS Cluster
echo "Step 1: Creating ECS Cluster..."
aws ecs create-cluster \
    --cluster-name hybrid-thesis-cluster \
    --capacity-providers FARGATE \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1

echo "✓ ECS Cluster created"
echo ""

# 2. Create CloudWatch Log Group
echo "Step 2: Creating CloudWatch Log Group..."
aws logs create-log-group \
    --log-group-name /ecs/hybrid-thesis-app 2>/dev/null || echo "  (Log group already exists)"

echo "✓ CloudWatch Log Group ready"
echo ""

# 3. Create ECS Task Execution Role
echo "Step 3: Creating ECS Task Execution Role..."
cat > ecs-task-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Check if role already exists
aws iam get-role --role-name ecsTaskExecutionRole 2>/dev/null
if [ $? -ne 0 ]; then
    aws iam create-role \
        --role-name ecsTaskExecutionRole \
        --assume-role-policy-document file://ecs-task-trust-policy.json

    aws iam attach-role-policy \
        --role-name ecsTaskExecutionRole \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    
    echo "✓ ECS Task Execution Role created"
else
    echo "✓ ECS Task Execution Role already exists"
fi

echo ""

# Wait for role to propagate
echo "Waiting for IAM role to propagate..."
sleep 10

# 4. Create Task Definitions for different resource configurations
echo "Step 4: Creating Task Definitions..."
create_task_definition() {
    local NAME=$1
    local CPU=$2
    local MEMORY=$3
    
    cat > task-def-${NAME}.json << EOF
{
  "family": "hybrid-thesis-${NAME}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "${CPU}",
  "memory": "${MEMORY}",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "${ECR_URI}:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hybrid-thesis-app",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "${NAME}"
        }
      }
    }
  ]
}
EOF
    
    aws ecs register-task-definition \
        --cli-input-json file://task-def-${NAME}.json > /dev/null
    
    echo "  ✓ Task definition created: hybrid-thesis-${NAME} (CPU: ${CPU}, Memory: ${MEMORY}MB)"
}

# Create task definitions matching Lambda configurations
# 128MB Lambda → 256 CPU (0.25 vCPU), 512MB memory
create_task_definition "lightweight" "256" "512"

# 256MB Lambda → 512 CPU (0.5 vCPU), 1024MB memory  
create_task_definition "thumbnail" "512" "1024"

# 512MB Lambda → 1024 CPU (1 vCPU), 2048MB memory
create_task_definition "medium" "1024" "2048"

# 512MB Lambda → 1024 CPU (1 vCPU), 2048MB memory
create_task_definition "heavy" "1024" "2048"

echo ""
echo "=========================================="
echo "✓ ECS Infrastructure Setup Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Cluster: hybrid-thesis-cluster"
echo "  - Region: ${AWS_REGION}"
echo "  - Task Definitions: 4 configurations created"
echo "  - ECR Image: ${ECR_URI}:latest"
echo ""
echo "Next Steps:"
echo "  1. Run setup-alb-and-services.sh to create the load balancer and services"
echo "  2. Wait 5-10 minutes for services to become healthy"
echo "  3. Test the endpoints"
echo ""