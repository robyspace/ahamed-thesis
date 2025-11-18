const fs = require('fs');
const path = require('path');

// ============================================
// PAYLOAD VARIANCE CONFIGURATION
// ============================================
// Define multiple payload sizes for each workload type
// to introduce meaningful variance in the experiment

const PAYLOAD_VARIANCE_CONFIGS = {
  lightweight_api: {
    sizes_kb: [0.5, 1, 5, 10],
    distribution: 'uniform',  // Equal probability
    baseContent: { user: 'test', action: 'validate' }
  },
  thumbnail_processing: {
    sizes_kb: [50, 100, 200, 500, 1024],
    distribution: 'uniform',
    generateBase64: true
  },
  medium_processing: {
    sizes_kb: [1024, 2048, 3072, 5120],
    distribution: 'uniform',
    generateBase64: true
  },
  heavy_processing: {
    sizes_kb: [3072, 5120, 8192, 10240],
    distribution: 'uniform',
    generateBase64: true
  }
};

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Generate unique request ID
 * @returns {string} Request ID in format req_timestamp_random
 */
function generateRequestId() {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Determine current time window based on hour of day
 * This helps track performance variance across different times
 * @returns {string} Time window identifier
 */
function getCurrentTimeWindow() {
  const hour = new Date().getHours();
  if (hour >= 2 && hour < 4) return 'early_morning';
  if (hour >= 8 && hour < 10) return 'morning_peak';
  if (hour >= 12 && hour < 14) return 'midday';
  if (hour >= 18 && hour < 20) return 'evening';
  if (hour >= 22 || hour < 1) return 'late_night';
  return 'other';
}

/**
 * Generate random string data of specified size
 * @param {number} sizeKB - Target size in kilobytes
 * @returns {string} Random data string
 */
function generateRandomData(sizeKB) {
  const sizeBytes = Math.floor(sizeKB * 1024);
  // Use a mix of characters for more realistic data
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < sizeBytes; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

/**
 * Generate base64-encoded data of specified size
 * Base64 encoding increases size by ~33%, so adjust input size accordingly
 * @param {number} targetSizeKB - Target size in kilobytes after base64 encoding
 * @returns {string} Base64-encoded data
 */
function generateBase64Data(targetSizeKB) {
  // Base64 encoding increases size by ~33%, so adjust input size
  const inputSizeKB = targetSizeKB / 1.33;
  const data = generateRandomData(inputSizeKB);
  return Buffer.from(data).toString('base64');
}

/**
 * Main payload generation function with variance
 * Randomly selects payload size from configured ranges
 * @param {string} workloadType - Type of workload (lightweight_api, thumbnail_processing, etc.)
 * @param {string} loadPattern - Load pattern identifier (low_load, medium_load, burst_load, ramp_load)
 * @returns {object} Complete payload with metadata
 */
function generateVariancePayload(workloadType, loadPattern) {
  const config = PAYLOAD_VARIANCE_CONFIGS[workloadType];

  if (!config) {
    throw new Error(`Unknown workload type: ${workloadType}`);
  }

  // Randomly select payload size from configured options
  const targetSizeKB = config.sizes_kb[
    Math.floor(Math.random() * config.sizes_kb.length)
  ];

  const requestId = generateRequestId();
  const timeWindow = getCurrentTimeWindow();
  const timestamp = new Date().toISOString();

  let payload = {
    request_id: requestId,
    workload_type: workloadType,
    timestamp: timestamp,
    metadata: {
      target_size_kb: targetSizeKB,
      time_window: timeWindow,
      load_pattern: loadPattern
    }
  };

  // Add workload-specific payload content
  if (workloadType === 'lightweight_api') {
    // For lightweight API, use JSON object with random data
    payload.payload = {
      ...config.baseContent,
      data: generateRandomData(targetSizeKB)
    };
  } else {
    // For processing workloads, use base64-encoded data
    // This simulates image/file processing workloads
    payload.payload = generateBase64Data(targetSizeKB);
  }

  return payload;
}

// ============================================
// ARTILLERY HOOK FUNCTIONS
// ============================================
// These functions are called by Artillery before each request
// They set the payload variable in the context

/**
 * Set variance payload for lightweight API workload
 * Artillery 2.0 signature: (userContext, events, done)
 */
function setLightweightVariancePayload(userContext, events, done) {
  try {
    const loadPattern = (userContext && userContext.vars && userContext.vars.load_pattern) || 'unknown';
    userContext.vars.payload = generateVariancePayload('lightweight_api', loadPattern);
  } catch (error) {
    console.error('Error in setLightweightVariancePayload:', error.message);
  }
  return done();
}

/**
 * Set variance payload for thumbnail processing workload
 * Artillery 2.0 signature: (userContext, events, done)
 */
function setThumbnailVariancePayload(userContext, events, done) {
  try {
    const loadPattern = (userContext && userContext.vars && userContext.vars.load_pattern) || 'unknown';
    userContext.vars.payload = generateVariancePayload('thumbnail_processing', loadPattern);
  } catch (error) {
    console.error('Error in setThumbnailVariancePayload:', error.message);
  }
  return done();
}

/**
 * Set variance payload for medium processing workload
 * Artillery 2.0 signature: (userContext, events, done)
 */
function setMediumVariancePayload(userContext, events, done) {
  try {
    const loadPattern = (userContext && userContext.vars && userContext.vars.load_pattern) || 'unknown';
    userContext.vars.payload = generateVariancePayload('medium_processing', loadPattern);
  } catch (error) {
    console.error('Error in setMediumVariancePayload:', error.message);
  }
  return done();
}

/**
 * Set variance payload for heavy processing workload
 * Artillery 2.0 signature: (userContext, events, done)
 */
function setHeavyVariancePayload(userContext, events, done) {
  try {
    const loadPattern = (userContext && userContext.vars && userContext.vars.load_pattern) || 'unknown';
    userContext.vars.payload = generateVariancePayload('heavy_processing', loadPattern);
  } catch (error) {
    console.error('Error in setHeavyVariancePayload:', error.message);
  }
  return done();
}

/**
 * Enhanced response logging with detailed metadata
 * Logs all request/response data to structured JSONL files
 * Artillery 2.0 signature: (requestParams, response, userContext, events, done)
 */
function logDetailedResponse(requestParams, response, userContext, events, done) {
  try {
    // Validate inputs
    if (!response || !response.body) {
      console.warn('No response body received');
      return done();
    }

    if (!userContext || !userContext.vars || !userContext.vars.payload) {
      console.warn('No payload found in context');
      return done();
    }

    const data = JSON.parse(response.body);
    const requestPayload = userContext.vars.payload;

    // Extract metadata with safe fallbacks
    const metadata = requestPayload.metadata || {};
    const timeWindow = metadata.time_window || 'unknown';
    const loadPattern = metadata.load_pattern || 'unknown';
    const targetSize = metadata.target_size_kb || 0;

    const logEntry = {
      request_id: data.request_id || requestPayload.request_id,
      platform: data.platform || 'unknown',
      workload_type: data.workload_type || requestPayload.workload_type || 'unknown',
      timestamp: data.timestamp || requestPayload.timestamp || new Date().toISOString(),
      http_status: response.statusCode,
      response_time_ms: response.timings ? response.timings.phases.total : null,

      // Enhanced metadata for variance analysis
      target_payload_size_kb: targetSize,
      actual_payload_size_kb: (data.metrics && data.metrics.payload_size_kb) || null,
      time_window: timeWindow,
      load_pattern: loadPattern,

      // Performance metrics from AWS
      metrics: data.metrics || {}
    };

    // Create log directory structure: data-output/variance-experiment/{time_window}_{load_pattern}/
    const date = new Date().toISOString().split('T')[0];
    const logDir = path.join(__dirname, '../data-output/variance-experiment', `${timeWindow}_${loadPattern}`);

    // Ensure directory exists
    if (!fs.existsSync(logDir)) {
      fs.mkdirSync(logDir, { recursive: true });
    }

    // Log file format: {platform}_{workload_type}_{date}.jsonl
    const platform = data.platform || 'unknown';
    const workloadType = (data.workload_type || requestPayload.workload_type || 'unknown').replace(/\s+/g, '_');
    const logFile = path.join(logDir, `${platform}_${workloadType}_${date}.jsonl`);

    fs.appendFileSync(logFile, JSON.stringify(logEntry) + '\n');

  } catch (e) {
    console.error('Error logging response:', e.message);
    console.error('Stack:', e.stack);
    // Log the error but don't fail the test
  }
  return done();
}

// ============================================
// EXPORTS
// ============================================
// Export all functions for use by Artillery

module.exports = {
  generateVariancePayload,
  setLightweightVariancePayload,
  setThumbnailVariancePayload,
  setMediumVariancePayload,
  setHeavyVariancePayload,
  logDetailedResponse,
  // Also export helper functions for testing
  getCurrentTimeWindow,
  generateRandomData,
  generateBase64Data,
  generateRequestId
};
