const fs = require('fs');
const path = require('path');

// Generate unique request ID
function generateRequestId() {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Generate payload based on workload type
function generatePayload(workloadType) {
  const requestId = generateRequestId();
  
  let payload = {
    request_id: requestId,
    workload_type: workloadType,
    timestamp: new Date().toISOString()
  };
  
  // Add workload-specific data
  switch(workloadType) {
    case 'lightweight_api':
      payload.payload = { user: 'test', action: 'validate', data: 'x'.repeat(1024) }; // ~1KB
      break;
    case 'thumbnail_processing':
      // Generate ~150KB base64 image data
      payload.payload = Buffer.from('x'.repeat(150 * 1024)).toString('base64');
      break;
    case 'medium_processing':
    payload.payload = Buffer.from('x'.repeat(2 * 1024 * 1024)).toString('base64'); // ~2.7 MB
    break;
    case 'heavy_processing':
    payload.payload = Buffer.from('x'.repeat(4 * 1024 * 1024)).toString('base64'); // ~5.3 MB
    break;
  }
  
  return payload;
}

// Set payload for lightweight requests
function setLightweightPayload(context, events, done) {
  // Ensure context.vars exists
  if (!context.vars) {
    context.vars = {};
  }
  context.vars.payload = generatePayload('lightweight_api');
  return done();
}

// Set payload for thumbnail requests
function setThumbnailPayload(context, events, done) {
  // Ensure context.vars exists
  if (!context.vars) {
    context.vars = {};
  }
  context.vars.payload = generatePayload('thumbnail_processing');
  return done();
}

// Set payload for medium requests
function setMediumPayload(context, events, done) {
  // Ensure context.vars exists
  if (!context.vars) {
    context.vars = {};
  }
  context.vars.payload = generatePayload('medium_processing');
  return done();
}

// Set payload for heavy requests
function setHeavyPayload(context, events, done) {
  // Ensure context.vars exists
  if (!context.vars) {
    context.vars = {};
  }
  context.vars.payload = generatePayload('heavy_processing');
  return done();
}

// Log response data
function logResponse(requestParams, response, context, ee, done) {
  if (response.body) {
    try {
      const data = typeof response.body === 'string' ? JSON.parse(response.body) : response.body;
      
      // Ensure data-output directory exists
      const outputDir = path.join(__dirname, '../data-output');
      if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir, { recursive: true });
      }
      
      const logEntry = {
        request_id: data.request_id || 'unknown',
        platform: data.platform || 'unknown',
        workload_type: data.workload_type || 'unknown',
        timestamp: data.timestamp || new Date().toISOString(),
        http_status: response.statusCode,
        response_time_ms: response.timings ? response.timings.phases.total : null,
        metrics: data.metrics || {}
      };
      
      // Create filename based on platform and date
      const date = new Date().toISOString().split('T')[0];
      const logFile = path.join(outputDir, `${data.platform}_${data.workload_type}_${date}.jsonl`);
      
      // Append to log file
      fs.appendFileSync(logFile, JSON.stringify(logEntry) + '\n');
    } catch (e) {
      console.error('Error logging response:', e.message);
    }
  }
  return done();
}

module.exports = {
  setLightweightPayload,
  setThumbnailPayload,
  setMediumPayload,
  setHeavyPayload,
  logResponse
};