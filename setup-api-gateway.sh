#!/bin/bash

set -e  # Exit on any error

# Configuration
export AWS_REGION="eu-west-1"  # Changed to your region
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=========================================="
echo "Hybrid Thesis API Gateway Setup"
echo "=========================================="
echo "AWS Region: $AWS_REGION"
echo "AWS Account ID: $AWS_ACCOUNT_ID"
echo ""

# Get API ID
export API_ID=$(aws apigateway get-rest-apis \
    --region $AWS_REGION \
    --query "items[?name=='HybridThesisAPI'].id" \
    --output text)

if [ -z "$API_ID" ] || [ "$API_ID" == "None" ]; then
    echo "ERROR: Could not find API 'HybridThesisAPI'"
    echo "Creating new API Gateway..."
    
    API_RESPONSE=$(aws apigateway create-rest-api \
        --region $AWS_REGION \
        --name "HybridThesisAPI" \
        --description "API for hybrid serverless-container research" \
        --endpoint-configuration types=REGIONAL)
    
    export API_ID=$(echo $API_RESPONSE | jq -r '.id')
    echo "Created new API with ID: $API_ID"
fi

# Get Root Resource ID
export ROOT_ID=$(aws apigateway get-resources \
    --region $AWS_REGION \
    --rest-api-id $API_ID \
    --query 'items[?path==`/`].id' \
    --output text)

echo "API ID: $API_ID"
echo "Root Resource ID: $ROOT_ID"
echo ""

# Check if Lambda functions exist
echo "Checking Lambda functions..."
LAMBDA_FUNCTIONS=(
    "hybrid-thesis-lightweight-api"
    "hybrid-thesis-thumbnail"
    "hybrid-thesis-medium"
    "hybrid-thesis-heavy"
)

MISSING_FUNCTIONS=()
for FUNC in "${LAMBDA_FUNCTIONS[@]}"; do
    if aws lambda get-function \
        --region $AWS_REGION \
        --function-name $FUNC \
        &>/dev/null; then
        echo "✓ Found function: $FUNC"
    else
        echo "✗ Missing function: $FUNC"
        MISSING_FUNCTIONS+=($FUNC)
    fi
done

if [ ${#MISSING_FUNCTIONS[@]} -gt 0 ]; then
    echo ""
    echo "ERROR: The following Lambda functions need to be created first:"
    for FUNC in "${MISSING_FUNCTIONS[@]}"; do
        echo "  - $FUNC"
    done
    echo ""
    echo "Would you like to continue without Lambda integration? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Exiting. Please create the Lambda functions first."
        exit 1
    fi
    SKIP_LAMBDA=true
else
    SKIP_LAMBDA=false
fi

echo ""
echo "Creating API endpoints..."
echo ""

# Function to create resource and method for each workload
create_endpoint() {
    local WORKLOAD=$1
    local FUNCTION_NAME=$2
    
    echo "----------------------------------------"
    echo "Creating endpoint: /$WORKLOAD"
    echo "----------------------------------------"
    
    # Check if resource already exists
    EXISTING_RESOURCE=$(aws apigateway get-resources \
        --region $AWS_REGION \
        --rest-api-id $API_ID \
        --query "items[?pathPart=='$WORKLOAD'].id" \
        --output text)
    
    if [ ! -z "$EXISTING_RESOURCE" ] && [ "$EXISTING_RESOURCE" != "None" ]; then
        echo "Resource /$WORKLOAD already exists with ID: $EXISTING_RESOURCE"
        RESOURCE_ID=$EXISTING_RESOURCE
    else
        # Create resource
        echo "Creating resource /$WORKLOAD..."
        RESOURCE_ID=$(aws apigateway create-resource \
            --region $AWS_REGION \
            --rest-api-id $API_ID \
            --parent-id $ROOT_ID \
            --path-part "$WORKLOAD" \
            --query 'id' \
            --output text)
        echo "Created resource with ID: $RESOURCE_ID"
    fi
    
    # Check if POST method already exists
    if aws apigateway get-method \
        --region $AWS_REGION \
        --rest-api-id $API_ID \
        --resource-id $RESOURCE_ID \
        --http-method POST \
        &>/dev/null; then
        echo "POST method already exists, deleting to recreate..."
        aws apigateway delete-method \
            --region $AWS_REGION \
            --rest-api-id $API_ID \
            --resource-id $RESOURCE_ID \
            --http-method POST || true
    fi
    
    # Create POST method
    echo "Creating POST method..."
    aws apigateway put-method \
        --region $AWS_REGION \
        --rest-api-id $API_ID \
        --resource-id $RESOURCE_ID \
        --http-method POST \
        --authorization-type NONE \
        --no-api-key-required
    
    if [ "$SKIP_LAMBDA" = false ]; then
        # Set up Lambda integration
        echo "Setting up Lambda integration..."
        aws apigateway put-integration \
            --region $AWS_REGION \
            --rest-api-id $API_ID \
            --resource-id $RESOURCE_ID \
            --http-method POST \
            --type AWS_PROXY \
            --integration-http-method POST \
            --uri "arn:aws:apigateway:${AWS_REGION}:lambda:path/2015-03-31/functions/arn:aws:lambda:${AWS_REGION}:${AWS_ACCOUNT_ID}:function:${FUNCTION_NAME}/invocations"
        
        # Add Lambda permission (remove existing first if it exists)
        echo "Adding Lambda permission..."
        aws lambda remove-permission \
            --region $AWS_REGION \
            --function-name $FUNCTION_NAME \
            --statement-id apigateway-access-$WORKLOAD \
            2>/dev/null || true
            
        aws lambda add-permission \
            --region $AWS_REGION \
            --function-name $FUNCTION_NAME \
            --statement-id apigateway-access-$WORKLOAD \
            --action lambda:InvokeFunction \
            --principal apigateway.amazonaws.com \
            --source-arn "arn:aws:execute-api:${AWS_REGION}:${AWS_ACCOUNT_ID}:${API_ID}/*/*"
    else
        # Set up mock integration for testing
        echo "Setting up mock integration (Lambda functions not available)..."
        aws apigateway put-integration \
            --region $AWS_REGION \
            --rest-api-id $API_ID \
            --resource-id $RESOURCE_ID \
            --http-method POST \
            --type MOCK \
            --request-templates '{"application/json": "{\"statusCode\": 200}"}'
        
        # Set up integration response
        aws apigateway put-integration-response \
            --region $AWS_REGION \
            --rest-api-id $API_ID \
            --resource-id $RESOURCE_ID \
            --http-method POST \
            --status-code 200 \
            --response-templates '{"application/json": "{\"message\": \"Mock response for '$WORKLOAD'\"}"}'
        
        # Set up method response
        aws apigateway put-method-response \
            --region $AWS_REGION \
            --rest-api-id $API_ID \
            --resource-id $RESOURCE_ID \
            --http-method POST \
            --status-code 200
    fi
    
    echo "✓ Created endpoint: /$WORKLOAD"
    echo ""
}

# Create endpoints for each workload type
create_endpoint "lightweight" "hybrid-thesis-lightweight-api"
create_endpoint "thumbnail" "hybrid-thesis-thumbnail"
create_endpoint "medium" "hybrid-thesis-medium"
create_endpoint "heavy" "hybrid-thesis-heavy"

# Deploy API
echo "=========================================="
echo "Deploying API to 'prod' stage..."
echo "=========================================="

aws apigateway create-deployment \
    --region $AWS_REGION \
    --rest-api-id $API_ID \
    --stage-name prod \
    --description "Deployment for hybrid thesis research"

echo ""
echo "=========================================="
echo "API Gateway setup complete!"
echo "=========================================="
echo ""
echo "Base URL: https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/prod"
echo ""
echo "Endpoints:"
echo "  - POST https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/prod/lightweight"
echo "  - POST https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/prod/thumbnail"
echo "  - POST https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/prod/medium"
echo "  - POST https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/prod/heavy"
echo ""

if [ "$SKIP_LAMBDA" = true ]; then
    echo "⚠️  WARNING: Lambda functions were not integrated."
    echo "    The endpoints are using mock responses."
    echo "    Create the Lambda functions and re-run this script to enable full functionality."
    echo ""
fi

echo "Test with:"
echo "curl -X POST https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/prod/lightweight \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"test\": \"data\"}'"
echo ""