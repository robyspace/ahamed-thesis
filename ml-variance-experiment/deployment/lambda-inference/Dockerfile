# Use AWS Lambda Python 3.9 base image for x86_64 architecture
FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.9

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY variance_model_best.pkl ${LAMBDA_TASK_ROOT}/
COPY feature_columns.json ${LAMBDA_TASK_ROOT}/
COPY model_metadata.json ${LAMBDA_TASK_ROOT}/

# Set the CMD to your handler
CMD [ "lambda_handler.lambda_handler" ]
