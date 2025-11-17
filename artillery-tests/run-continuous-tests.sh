#!/bin/bash

# Configuration - MODIFY THESE FOR YOUR NEEDS
TEST_MODE="full"  # Options: "short" (few hours), "full" (10 days)
SHORT_TEST_HOURS=3  # For quick testing
FULL_TEST_DAYS=0.25   # For full data collection

API_GATEWAY_URL="jt67vt5uwj.execute-api.eu-west-1.amazonaws.com/prod"
ALB_DNS="hybrid-thesis-alb-811686247.eu-west-1.elb.amazonaws.com"

# Create output directory
mkdir -p ../data-output-full

# Determine test duration
if [ "$TEST_MODE" = "short" ]; then
    TOTAL_ITERATIONS=$SHORT_TEST_HOURS
    SLEEP_DURATION=360  # 6 mins between iterations
    echo "Running in SHORT TEST MODE: $SHORT_TEST_HOURS iterations (hours)"
else
    TOTAL_ITERATIONS=$((24 * FULL_TEST_DAYS))
    SLEEP_DURATION=600  # 10 mins between iterations
    echo "Running in FULL MODE: $FULL_TEST_DAYS days"
fi

echo "Starting data collection..."
echo "Lambda URL: https://$API_GATEWAY_URL"
echo "ECS URL: http://$ALB_DNS"
echo "Total iterations: $TOTAL_ITERATIONS"
echo ""

# Update Artillery config with actual URLs
sed -i '' "s|YOUR_API_ID.execute-api.eu-west-1.amazonaws.com/prod|$API_GATEWAY_URL|g" dual-platform-test.yml
sed -i '' "s|YOUR_ALB_DNS|$ALB_DNS|g" dual-platform-test.yml

# Function to run a single test iteration
run_test_iteration() {
    local iteration=$1
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    echo "[$(date)] Starting iteration $iteration of $TOTAL_ITERATIONS..."
    
    artillery run dual-platform-test.yml \
        --output ../data-output/artillery_report_${timestamp}.json
    
    echo "[$(date)] Iteration $iteration complete"
}

# Run tests
for i in $(seq 1 $TOTAL_ITERATIONS); do
    run_test_iteration $i
    
    # Sleep before next iteration (except on last one)
    if [ $i -lt $TOTAL_ITERATIONS ]; then
        echo "Sleeping for 1 hour..."
        sleep $SLEEP_DURATION
    fi
done

echo ""
echo "Data collection complete!"
echo "Total iterations completed: $TOTAL_ITERATIONS"
echo "Data files location: ../data-output-full/"