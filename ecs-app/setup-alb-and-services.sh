#!/bin/bash

export AWS_REGION="eu-west-1"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Setting up Application Load Balancer and ECS Services..."

# 1. Get default VPC and subnets
export VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=isDefault,Values=true" \
    --query 'Vpcs[0].VpcId' \
    --output text)

export SUBNET_IDS=$(aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=${VPC_ID}" \
    --query 'Subnets[*].SubnetId' \
    --output text | tr '\t' ',')

echo "Using VPC: $VPC_ID"
echo "Using Subnets: $SUBNET_IDS"

# 2. Create Security Group for ALB
export ALB_SG=$(aws ec2 create-security-group \
    --group-name hybrid-thesis-alb-sg \
    --description "Security group for Hybrid Thesis ALB" \
    --vpc-id $VPC_ID \
    --query 'GroupId' \
    --output text)

aws ec2 authorize-security-group-ingress \
    --group-id $ALB_SG \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

echo "✓ ALB Security Group created: $ALB_SG"

# 3. Create Security Group for ECS Tasks
export ECS_SG=$(aws ec2 create-security-group \
    --group-name hybrid-thesis-ecs-sg \
    --description "Security group for Hybrid Thesis ECS Tasks" \
    --vpc-id $VPC_ID \
    --query 'GroupId' \
    --output text)

aws ec2 authorize-security-group-ingress \
    --group-id $ECS_SG \
    --protocol tcp \
    --port 8080 \
    --source-group $ALB_SG

echo "✓ ECS Security Group created: $ECS_SG"

# 4. Create Application Load Balancer
export ALB_ARN=$(aws elbv2 create-load-balancer \
    --name hybrid-thesis-alb \
    --subnets $(echo $SUBNET_IDS | tr ',' ' ') \
    --security-groups $ALB_SG \
    --scheme internet-facing \
    --type application \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text)

export ALB_DNS=$(aws elbv2 describe-load-balancers \
    --load-balancer-arns $ALB_ARN \
    --query 'LoadBalancers[0].DNSName' \
    --output text)

echo "✓ ALB created: $ALB_DNS"

# 5. Create Target Groups for each workload type
create_target_group() {
    local NAME=$1
    
    TG_ARN=$(aws elbv2 create-target-group \
        --name hybrid-thesis-${NAME}-tg \
        --protocol HTTP \
        --port 8080 \
        --vpc-id $VPC_ID \
        --target-type ip \
        --health-check-path /health \
        --health-check-interval-seconds 30 \
        --health-check-timeout-seconds 5 \
        --healthy-threshold-count 2 \
        --unhealthy-threshold-count 3 \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text)
    
    echo $TG_ARN
}

export TG_LIGHTWEIGHT=$(create_target_group "lightweight")
export TG_THUMBNAIL=$(create_target_group "thumbnail")
export TG_MEDIUM=$(create_target_group "medium")
export TG_HEAVY=$(create_target_group "heavy")

echo "✓ Target Groups created"

# 6. Create ALB Listener and Rules
export LISTENER_ARN=$(aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port 80 \
    --default-actions Type=forward,TargetGroupArn=$TG_LIGHTWEIGHT \
    --query 'Listeners[0].ListenerArn' \
    --output text)

# Add path-based routing rules
aws elbv2 create-rule \
    --listener-arn $LISTENER_ARN \
    --priority 1 \
    --conditions Field=path-pattern,Values='/lightweight*' \
    --actions Type=forward,TargetGroupArn=$TG_LIGHTWEIGHT

aws elbv2 create-rule \
    --listener-arn $LISTENER_ARN \
    --priority 2 \
    --conditions Field=path-pattern,Values='/thumbnail*' \
    --actions Type=forward,TargetGroupArn=$TG_THUMBNAIL

aws elbv2 create-rule \
    --listener-arn $LISTENER_ARN \
    --priority 3 \
    --conditions Field=path-pattern,Values='/medium*' \
    --actions Type=forward,TargetGroupArn=$TG_MEDIUM

aws elbv2 create-rule \
    --listener-arn $LISTENER_ARN \
    --priority 4 \
    --conditions Field=path-pattern,Values='/heavy*' \
    --actions Type=forward,TargetGroupArn=$TG_HEAVY

echo "✓ ALB Listener and Rules configured"

# 7. Create ECS Services
create_ecs_service() {
    local NAME=$1
    local TG_ARN=$2
    local DESIRED_COUNT=$3
    
    aws ecs create-service \
        --cluster hybrid-thesis-cluster \
        --service-name hybrid-thesis-${NAME}-service \
        --task-definition hybrid-thesis-${NAME} \
        --desired-count $DESIRED_COUNT \
        --launch-type FARGATE \
        --network-configuration "awsvpcConfiguration={subnets=[$(echo $SUBNET_IDS | sed 's/,/\",\"/g' | sed 's/^/\"/' | sed 's/$/\"/')],securityGroups=[$ECS_SG],assignPublicIp=ENABLED}" \
        --load-balancers "targetGroupArn=${TG_ARN},containerName=app,containerPort=8080"
    
    echo "✓ ECS Service created: hybrid-thesis-${NAME}-service"
}

# Create services with minimal instances
create_ecs_service "lightweight" $TG_LIGHTWEIGHT 1
create_ecs_service "thumbnail" $TG_THUMBNAIL 1
create_ecs_service "medium" $TG_MEDIUM 1
create_ecs_service "heavy" $TG_HEAVY 1

echo ""
echo "======================================"
echo "ECS Infrastructure Setup Complete!"
echo "======================================"
echo ""
echo "Application Load Balancer DNS: $ALB_DNS"
echo ""
echo "ECS Endpoints:"
echo "  - http://${ALB_DNS}/lightweight/process"
echo "  - http://${ALB_DNS}/thumbnail/process"
echo "  - http://${ALB_DNS}/medium/process"
echo "  - http://${ALB_DNS}/heavy/process"
echo ""
echo "Health check: http://${ALB_DNS}/health"
echo ""
echo "Wait 5-10 minutes for services to become healthy before testing"