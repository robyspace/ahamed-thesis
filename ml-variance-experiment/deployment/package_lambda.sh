#!/bin/bash
# Package Lambda function with sklearn bundled directly
# This avoids the layer size limit

set -e

echo "Packaging Lambda function with sklearn..."

cd lambda-inference

# Clean up old package
rm -rf package
rm -f function.zip

# Create package directory
mkdir package

# Install sklearn and scipy using Docker for correct platform
echo "Installing scikit-learn and scipy..."
docker run --platform linux/amd64 -v "$PWD":/var/task "public.ecr.aws/sam/build-python3.9:latest" /bin/sh -c "
    pip install scikit-learn==1.3.2 scipy -t /var/task/package/
"

# Copy Lambda handler and supporting files
echo "Copying function code and model files..."
cp lambda_handler.py package/
cp variance_model_best.pkl package/
cp feature_columns.json package/
cp model_metadata.json package/

# Create zip file
echo "Creating deployment package..."
cd package
zip -r9 ../function.zip .
cd ..

echo ""
echo "Lambda function packaged successfully!"
echo "function.zip size: $(du -h function.zip | cut -f1)"
echo ""
echo "To deploy, run:"
echo "  aws lambda update-function-code --function-name hybrid-ml-router --zip-file fileb://function.zip"
