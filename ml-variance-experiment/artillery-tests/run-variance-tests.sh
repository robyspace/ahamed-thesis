#!/bin/bash

set -e

echo "=========================================="
echo "ML VARIANCE EXPERIMENT - DATA COLLECTION"
echo "=========================================="
echo ""

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data-output/variance-experiment"
LOGS_DIR="${SCRIPT_DIR}/../logs"

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p "${LOGS_DIR}"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to run a test
run_test() {
  local test_name=$1
  local config_file=$2
  local time_window=$3

  echo ""
  echo "=========================================="
  echo -e "${BLUE}Running: ${test_name}${NC}"
  echo "Time: $(date)"
  echo "Config: ${config_file}"
  echo "Expected window: ${time_window}"
  echo "=========================================="
  echo ""

  # Run artillery test
  cd "${SCRIPT_DIR}"

  if [ -f "${config_file}" ]; then
    echo -e "${GREEN}Starting Artillery test...${NC}"
    artillery run "${config_file}"

    echo ""
    echo -e "${GREEN}✓ Completed: ${test_name}${NC}"
    echo ""
  else
    echo -e "${RED}✗ Error: Config file not found: ${config_file}${NC}"
    exit 1
  fi
}

# Display schedule
echo -e "${BLUE} TEST SCHEDULE:${NC}"
echo ""
echo "1. Early Morning (1-4 AM)   → Low Load      (variance-test-low-load.yml)"
echo "2. Morning Peak (7-10 AM)    → Ramp Load     (variance-test-ramp-load.yml)"
echo "3. Midday (12-2 PM)         → Medium Load   (variance-test-medium-load.yml)"
echo "4. Evening (5-7 PM)         → Burst Load    (variance-test-burst-load.yml)"
echo "5. Late Night (10-12 PM)    → Low Load      (variance-test-low-load.yml)"
echo ""
echo -e "${YELLOW}  IMPORTANT: Run each test during its designated time window!${NC}"
echo ""

# Check current time and suggest which test to run
current_hour=$(date +%H)
current_hour=$((10#$current_hour))  # Remove leading zero

suggested_test=""
suggested_name=""
suggested_window=""

if [ $current_hour -ge 1 ] && [ $current_hour -lt 4 ]; then
  suggested_test="variance-test-low-load.yml"
  suggested_name="Test 1: Early Morning - Low Load"
  suggested_window="early_morning"
  echo -e "${GREEN} Current time suggests: Test 1 (Early Morning - Low Load)${NC}"
elif [ $current_hour -ge 7 ] && [ $current_hour -lt 10 ]; then
  suggested_test="variance-test-ramp-load.yml"
  suggested_name="Test 2: Morning Peak - Ramp Load"
  suggested_window="morning_peak"
  echo -e "${GREEN} Current time suggests: Test 2 (Morning Peak - Ramp Load)${NC}"
elif [ $current_hour -ge 12 ] && [ $current_hour -lt 14 ]; then
  suggested_test="variance-test-medium-load.yml"
  suggested_name="Test 3: Midday - Medium Load"
  suggested_window="midday"
  echo -e "${GREEN} Current time suggests: Test 3 (Midday - Medium Load)${NC}"
elif [ $current_hour -ge 17 ] && [ $current_hour -lt 19 ]; then
  suggested_test="variance-test-burst-load.yml"
  suggested_name="Test 4: Evening - Burst Load"
  suggested_window="evening"
  echo -e "${GREEN} Current time suggests: Test 4 (Evening - Burst Load)${NC}"
elif [ $current_hour -ge 22 ] || [ $current_hour -lt 24 ]; then
  suggested_test="variance-test-low-load.yml"
  suggested_name="Test 5: Late Night - Low Load"
  suggested_window="late_night"
  echo -e "${GREEN} Current time suggests: Test 5 (Late Night - Low Load)${NC}"
else
  echo -e "${YELLOW}  Current time (${current_hour}:00) is not in a designated test window${NC}"
  echo "    Please run tests during scheduled windows for best variance"
  echo ""
  echo "Available test files:"
  echo "  - variance-test-low-load.yml"
  echo "  - variance-test-medium-load.yml"
  echo "  - variance-test-burst-load.yml"
  echo "  - variance-test-ramp-load.yml"
  echo ""
  echo "Usage: ./run-variance-tests.sh [config-file]"
  echo "   or: artillery run <config-file>"
  exit 0
fi

echo ""

# Check if manual config specified
if [ -n "$1" ]; then
  manual_config="$1"
  echo -e "${YELLOW}Manual config specified: ${manual_config}${NC}"
  echo ""
  read -p "Press Enter to start test, or Ctrl+C to cancel..."
  run_test "Manual Test" "${manual_config}" "manual"
elif [ -n "$suggested_test" ]; then
  echo -e "${GREEN}Recommended test: ${suggested_test}${NC}"
  echo ""
  read -p "Press Enter to start test, or Ctrl+C to cancel..."
  run_test "${suggested_name}" "${suggested_test}" "${suggested_window}"
fi

# Post-test summary
echo ""
echo "=========================================="
echo -e "${GREEN}✓ Test completed successfully!${NC}"
echo "=========================================="
echo ""
echo "Data saved to: ${DATA_DIR}"
echo ""
echo "Quick validation commands:"
echo ""
echo "# Check sample count:"
echo "  wc -l ${DATA_DIR}/*/*.jsonl"
echo ""
echo "# Inspect first log entry:"
echo "  head -1 ${DATA_DIR}/*/*.jsonl | jq ."
echo ""
echo "# Check unique payload sizes:"
echo "  cat ${DATA_DIR}/*/*.jsonl | jq -r '.target_payload_size_kb' | sort -u"
echo ""
echo "=========================================="
