import { check, sleep } from 'k6';
import http from 'k6/http';
import { Counter, Gauge, Rate, Trend } from 'k6/metrics';

// Custom metrics for comprehensive monitoring
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');
const memoryDuration = new Trend('memory_request_duration');
const basicDuration = new Trend('basic_request_duration');
const requestsPerStage = new Counter('requests_per_stage');
const currentLoad = new Gauge('current_load_level');
const requestTypeDistribution = new Counter('request_type_distribution');

// RL Autoscaler Training Test - Comprehensive Real-World Scenarios
// This test simulates realistic traffic patterns to train the RL agent effectively
//
// CUSTOMIZABLE DURATION:
// Set DURATION_MULTIPLIER environment variable to extend test duration
// Examples:
//   DURATION_MULTIPLIER=1    -> 60 minutes (default, 1 hour)
//   DURATION_MULTIPLIER=24   -> 24 hours (1 day)
//   DURATION_MULTIPLIER=48   -> 48 hours (2 days)
//   DURATION_MULTIPLIER=168  -> 168 hours (1 week)
//
// Set CYCLE_COUNT to repeat the pattern multiple times
// Examples:
//   CYCLE_COUNT=1   -> Run pattern once (default)
//   CYCLE_COUNT=10  -> Repeat pattern 10 times
//   CYCLE_COUNT=100 -> Repeat pattern 100 times (useful for multi-day training)
//
// Usage:
//   k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=7 k6-autoscaler-training.js
//   This runs the pattern for 7 cycles, each cycle taking 24 hours = 1 week total

const DURATION_MULTIPLIER = parseFloat(__ENV.DURATION_MULTIPLIER || '1');
const CYCLE_COUNT = parseInt(__ENV.CYCLE_COUNT || '1');

// Helper function to scale duration
function scaleDuration(minutes) {
  const totalMinutes = minutes * DURATION_MULTIPLIER;
  const hours = Math.floor(totalMinutes / 60);
  const mins = Math.floor(totalMinutes % 60);
  const secs = Math.floor((totalMinutes % 1) * 60);

  if (hours > 0) {
    return mins > 0 ? `${hours}h${mins}m` : `${hours}h`;
  } else if (mins > 0) {
    return secs > 0 ? `${mins}m${secs}s` : `${mins}m`;
  } else {
    return `${secs}s`;
  }
}

// Base pattern (1 hour cycle)
const basePattern = [
  // ===== WARM-UP PHASE (Baseline establishment) =====
  { duration: scaleDuration(1), target: 0 },     // No requests initially
  { duration: scaleDuration(1), target: 2 },     // Gentle start - 2 users
  { duration: scaleDuration(2), target: 2 },     // Baseline LOW traffic

  // ===== PHASE 1: GRADUAL MORNING RAMP-UP (Simulates business hours start) =====
  { duration: scaleDuration(1), target: 5 },     // Early morning users arrive
  { duration: scaleDuration(2), target: 5 },     // Low morning traffic
  { duration: scaleDuration(1), target: 10 },    // More users logging in
  { duration: scaleDuration(2), target: 10 },    // Growing morning traffic
  { duration: scaleDuration(1), target: 15 },    // Peak morning traffic
  { duration: scaleDuration(3), target: 15 },    // Sustained morning activity

  // ===== PHASE 2: STEADY DAYTIME LOAD (Normal business operations) =====
  { duration: scaleDuration(1), target: 20 },    // Midday increase
  { duration: scaleDuration(4), target: 20 },    // Stable daytime load

  // ===== PHASE 3: LUNCH HOUR DIP (Realistic traffic pattern) =====
  { duration: scaleDuration(1), target: 12 },    // Users taking lunch break
  { duration: scaleDuration(2), target: 12 },    // Reduced lunch activity

  // ===== PHASE 4: POST-LUNCH RECOVERY =====
  { duration: scaleDuration(1), target: 18 },    // Users returning
  { duration: scaleDuration(3), target: 18 },    // Afternoon steady state

  // ===== PHASE 5: AFTERNOON PEAK (Highest daily load) =====
  { duration: scaleDuration(1), target: 25 },    // Building to peak
  { duration: scaleDuration(2), target: 30 },    // Ramp to peak
  { duration: scaleDuration(4), target: 30 },    // Sustained peak load

  // ===== PHASE 6: FLASH SPIKE (Sudden event - viral content, promotion) =====
  { duration: scaleDuration(0.5), target: 50 },  // Sudden viral spike
  { duration: scaleDuration(2), target: 50 },    // Spike sustained
  { duration: scaleDuration(1), target: 30 },    // Quick recovery to normal peak

  // ===== PHASE 7: GRADUAL EVENING DECLINE =====
  { duration: scaleDuration(1), target: 25 },    // Early evening decrease
  { duration: scaleDuration(2), target: 20 },    // Continued decline
  { duration: scaleDuration(2), target: 15 },    // Further decrease
  { duration: scaleDuration(2), target: 10 },    // Late evening

  // ===== PHASE 8: NIGHT-TIME LOW (Maintenance window simulation) =====
  { duration: scaleDuration(1), target: 5 },     // Night users
  { duration: scaleDuration(3), target: 5 },     // Sustained low load
  { duration: scaleDuration(1), target: 3 },     // Deep night
  { duration: scaleDuration(2), target: 3 },     // Minimal activity

  // ===== PHASE 9: OSCILLATING LOAD (Test rapid adaptation) =====
  { duration: scaleDuration(0.5), target: 15 },  // Quick up
  { duration: scaleDuration(1), target: 15 },    // Hold
  { duration: scaleDuration(0.5), target: 8 },   // Quick down
  { duration: scaleDuration(1), target: 8 },     // Hold
  { duration: scaleDuration(0.5), target: 20 },  // Quick up again
  { duration: scaleDuration(1), target: 20 },    // Hold
  { duration: scaleDuration(0.5), target: 5 },   // Quick down

  // ===== PHASE 10: GRACEFUL SHUTDOWN =====
  { duration: scaleDuration(1), target: 2 },     // Final users
  { duration: scaleDuration(0.5), target: 0 },   // Complete ramp down
];

// Generate stages by repeating the pattern
function generateStages() {
  let stages = [];
  for (let i = 0; i < CYCLE_COUNT; i++) {
    stages = stages.concat(basePattern);
  }
  return stages;
}

export const options = {
  stages: generateStages(),
  thresholds: {
    http_req_duration: ['p(95)<8000'],   // 95% under 8s (realistic for intensive ops)
    'http_req_duration{expected_response:true}': ['p(99)<12000'],  // 99% under 12s
    errors: ['rate<0.60'],                // Max 60% error rate (relaxed for RL training phase)
    'cpu_request_duration': ['p(95)<10000'],  // Relaxed to 10s for heavy CPU workload
    'memory_request_duration': ['p(95)<4000'],
    'basic_request_duration': ['p(95)<1000'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';

// Request type distribution that mimics real-world application usage
const REQUEST_PATTERNS = {
  LIGHT: { cpu: 0.10, memory: 0.05, basic: 0.85 },      // Night-time pattern
  NORMAL: { cpu: 0.30, memory: 0.20, basic: 0.50 },     // Regular business hours
  INTENSIVE: { cpu: 0.45, memory: 0.35, basic: 0.20 },  // Peak hours pattern
  SPIKE: { cpu: 0.40, memory: 0.40, basic: 0.20 },      // Viral/spike pattern
};

// Determine current load phase based on VU count
function getLoadPhase(vu) {
  if (vu <= 3) return 'NIGHT';
  if (vu <= 10) return 'LOW';
  if (vu <= 20) return 'MEDIUM';
  if (vu <= 30) return 'HIGH';
  if (vu <= 50) return 'PEAK';
  return 'EXTREME';
}

// Get request pattern based on load phase
function getRequestPattern(phase) {
  switch(phase) {
    case 'NIGHT':
      return REQUEST_PATTERNS.LIGHT;
    case 'LOW':
    case 'MEDIUM':
      return REQUEST_PATTERNS.NORMAL;
    case 'HIGH':
      return REQUEST_PATTERNS.INTENSIVE;
    case 'PEAK':
    case 'EXTREME':
      return REQUEST_PATTERNS.SPIKE;
    default:
      return REQUEST_PATTERNS.NORMAL;
  }
}

// Calculate sleep time based on load (inverse relationship)
function calculateSleepTime(phase, requestType) {
  let baseSleep;

  switch(phase) {
    case 'NIGHT':
      baseSleep = 4.0;   // Very long sleep between requests
      break;
    case 'LOW':
      baseSleep = 2.5;   // Long sleep
      break;
    case 'MEDIUM':
      baseSleep = 1.5;   // Moderate sleep
      break;
    case 'HIGH':
      baseSleep = 0.8;   // Short sleep
      break;
    case 'PEAK':
      baseSleep = 0.4;   // Very short sleep
      break;
    case 'EXTREME':
      baseSleep = 0.2;   // Minimal sleep (high pressure)
      break;
    default:
      baseSleep = 1.0;
  }

  // Basic requests can be faster, intensive requests need more spacing
  const typeMultiplier = requestType === 'basic' ? 0.7 : 1.0;

  // Add randomness to simulate realistic user behavior
  const randomFactor = 0.5 + Math.random() * 1.0; // 0.5x to 1.5x

  return baseSleep * typeMultiplier * randomFactor;
}

// Generate CPU iterations based on load phase and realism
// Optimized for 500m CPU limit with MAX_CPU_ITERATIONS=2500000
function getCpuIterations(phase) {
  let base, variance;

  switch(phase) {
    case 'NIGHT':
    case 'LOW':
      base = 500000;      // Light CPU tasks (0.5M-1M iterations)
      variance = 500000;  // ~1-3 seconds CPU time
      break;
    case 'MEDIUM':
      base = 1000000;     // Moderate CPU tasks (1M-1.5M iterations)
      variance = 500000;  // ~2-5 seconds CPU time
      break;
    case 'HIGH':
      base = 1500000;     // Heavy CPU tasks (1.5M-2M iterations)
      variance = 500000;  // ~4-8 seconds CPU time
      break;
    case 'PEAK':
    case 'EXTREME':
      base = 2000000;     // Maximum CPU tasks (2M-2.5M iterations)
      variance = 500000;  // ~6-12 seconds CPU time (safe under concurrent load)
      break;
    default:
      base = 1200000;
      variance = 500000;
  }

  return Math.floor(base + Math.random() * variance);
}

// Generate memory size based on load phase and realism
// Optimized for 512Mi container limit with MAX_MEMORY_MB=140
function getMemorySize(phase) {
  let base, variance;

  switch(phase) {
    case 'NIGHT':
    case 'LOW':
      base = 20;          // Small allocations (20-35 MB)
      variance = 15;
      break;
    case 'MEDIUM':
      base = 40;          // Moderate allocations (40-60 MB)
      variance = 20;
      break;
    case 'HIGH':
      base = 60;          // Large allocations (60-85 MB)
      variance = 25;
      break;
    case 'PEAK':
    case 'EXTREME':
      base = 80;          // Maximum allocations (80-120 MB)
      variance = 40;      // Up to 120 MB max (safe for 140 MB limit)
      break;
    default:
      base = 50;
      variance = 20;
  }

  return Math.floor(base + Math.random() * variance);
}

export default function () {
  const phase = getLoadPhase(__VU);
  const pattern = getRequestPattern(phase);

  // Update gauge metric for monitoring
  currentLoad.add(__VU);

  // Determine request type based on pattern distribution
  const rand = Math.random();
  let requestType;

  if (rand < pattern.cpu) {
    requestType = 'cpu';
  } else if (rand < pattern.cpu + pattern.memory) {
    requestType = 'memory';
  } else {
    requestType = 'basic';
  }

  // Track request type distribution
  requestTypeDistribution.add(1, { type: requestType, phase: phase });

  // Execute request based on type
  let res;

  switch(requestType) {
    case 'cpu':
      const iterations = getCpuIterations(phase);
      res = http.get(`${BASE_URL}/api/cpu?iterations=${iterations}`, {
        tags: { request_type: 'cpu', load_phase: phase },
        timeout: '30s', // Prevent hanging requests
      });

      cpuDuration.add(res.timings.duration);

      check(res, {
        'cpu api status is 200': (r) => r.status === 200,
        'cpu api completed': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.status === 'success';
          } catch (e) {
            return false;
          }
        },
        'cpu api response time acceptable': (r) => r.timings.duration < 10000,
      }) || errorRate.add(1);
      break;

    case 'memory':
      const sizeMb = getMemorySize(phase);
      res = http.get(`${BASE_URL}/api/memory?size_mb=${sizeMb}`, {
        tags: { request_type: 'memory', load_phase: phase },
        timeout: '20s',
      });

      memoryDuration.add(res.timings.duration);

      check(res, {
        'memory api status is 200': (r) => r.status === 200,
        'memory api completed': (r) => {
          try {
            const body = JSON.parse(r.body);
            return body.status === 'success';
          } catch (e) {
            return false;
          }
        },
        'memory api response time acceptable': (r) => r.timings.duration < 8000,
      }) || errorRate.add(1);
      break;

    case 'basic':
      res = http.get(`${BASE_URL}/api`, {
        tags: { request_type: 'basic', load_phase: phase },
        timeout: '5s',
      });

      basicDuration.add(res.timings.duration);

      check(res, {
        'basic api status is 200': (r) => r.status === 200,
        'basic api response time acceptable': (r) => r.timings.duration < 2000,
      }) || errorRate.add(1);
      break;
  }

  // Track request count
  requestsPerStage.add(1, { phase: phase });

  // Realistic sleep pattern based on load and request type
  const sleepTime = calculateSleepTime(phase, requestType);
  sleep(sleepTime);
}

// Enhanced summary with training-specific insights
export function handleSummary(data) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
  const timeHour = new Date().toISOString().replace(/[:.]/g, '-').split('T')[1].substring(0, 5);

  return {
    [`training-summary-${timestamp}-${timeHour}.json`]: JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options = {}) {
  const indent = options.indent || '';

  let summary = '\n' + indent + '🎯 RL AUTOSCALER TRAINING TEST - COMPREHENSIVE RESULTS\n';
  summary += indent + '═══════════════════════════════════════════════════════\n\n';

  // Test Configuration
  summary += indent + '⚙️  TEST CONFIGURATION\n';
  summary += indent + `   Duration Multiplier: ${DURATION_MULTIPLIER}x\n`;
  summary += indent + `   Cycle Count: ${CYCLE_COUNT}\n`;
  summary += indent + `   Base Pattern: 60 minutes → Actual: ${(60 * DURATION_MULTIPLIER).toFixed(1)} min per cycle\n`;
  summary += indent + `   Total Planned Duration: ${(60 * DURATION_MULTIPLIER * CYCLE_COUNT / 60).toFixed(1)} hours (${(60 * DURATION_MULTIPLIER * CYCLE_COUNT / 60 / 24).toFixed(1)} days)\n\n`;

  // Test Overview
  const duration = (data.state?.testRunDurationMs / 1000 / 60) || 0;
  summary += indent + '⏱️  TEST DURATION\n';
  summary += indent + `   Total Runtime: ${duration.toFixed(1)} minutes (${(duration/60).toFixed(1)} hours / ${(duration/60/24).toFixed(2)} days)\n`;
  summary += indent + `   Start Time: ${new Date(data.state?.testRunDurationMs ? Date.now() - data.state.testRunDurationMs : Date.now()).toLocaleString()}\n\n`;

  // Request Statistics
  const totalRequests = data.metrics.http_reqs?.values.count || 0;
  const failedRate = data.metrics.http_req_failed?.values.rate || 0;
  const successRate = (1 - failedRate) * 100;

  summary += indent + '📊 REQUEST STATISTICS\n';
  summary += indent + `   Total Requests: ${totalRequests.toLocaleString()}\n`;
  summary += indent + `   Successful: ${Math.floor(totalRequests * (1 - failedRate)).toLocaleString()} (${successRate.toFixed(2)}%)\n`;
  summary += indent + `   Failed: ${Math.floor(totalRequests * failedRate).toLocaleString()} (${(failedRate * 100).toFixed(2)}%)\n`;
  summary += indent + `   Requests/sec: ${(totalRequests / (duration * 60)).toFixed(2)}\n\n`;

  // Response Time Analysis
  summary += indent + '⚡ RESPONSE TIME METRICS\n';
  summary += indent + `   Average: ${(data.metrics.http_req_duration?.values.avg || 0).toFixed(0)}ms\n`;
  summary += indent + `   Median (p50): ${(data.metrics.http_req_duration?.values.med || 0).toFixed(0)}ms\n`;
  summary += indent + `   90th percentile: ${(data.metrics.http_req_duration?.values['p(90)'] || 0).toFixed(0)}ms\n`;
  summary += indent + `   95th percentile: ${(data.metrics.http_req_duration?.values['p(95)'] || 0).toFixed(0)}ms\n`;
  summary += indent + `   99th percentile: ${(data.metrics.http_req_duration?.values['p(99)'] || 0).toFixed(0)}ms\n`;
  summary += indent + `   Maximum: ${(data.metrics.http_req_duration?.values.max || 0).toFixed(0)}ms\n\n`;

  // Endpoint-Specific Performance
  if (data.metrics.cpu_request_duration) {
    summary += indent + '🔥 CPU ENDPOINT PERFORMANCE\n';
    summary += indent + `   Avg Duration: ${(data.metrics.cpu_request_duration.values.avg || 0).toFixed(0)}ms\n`;
    summary += indent + `   p95 Duration: ${(data.metrics.cpu_request_duration.values['p(95)'] || 0).toFixed(0)}ms\n`;
    summary += indent + `   p99 Duration: ${(data.metrics.cpu_request_duration.values['p(99)'] || 0).toFixed(0)}ms\n`;
    summary += indent + `   Max Duration: ${(data.metrics.cpu_request_duration.values.max || 0).toFixed(0)}ms\n\n`;
  }

  if (data.metrics.memory_request_duration) {
    summary += indent + '💾 MEMORY ENDPOINT PERFORMANCE\n';
    summary += indent + `   Avg Duration: ${(data.metrics.memory_request_duration.values.avg || 0).toFixed(0)}ms\n`;
    summary += indent + `   p95 Duration: ${(data.metrics.memory_request_duration.values['p(95)'] || 0).toFixed(0)}ms\n`;
    summary += indent + `   p99 Duration: ${(data.metrics.memory_request_duration.values['p(99)'] || 0).toFixed(0)}ms\n`;
    summary += indent + `   Max Duration: ${(data.metrics.memory_request_duration.values.max || 0).toFixed(0)}ms\n\n`;
  }

  if (data.metrics.basic_request_duration) {
    summary += indent + '📦 BASIC ENDPOINT PERFORMANCE\n';
    summary += indent + `   Avg Duration: ${(data.metrics.basic_request_duration.values.avg || 0).toFixed(0)}ms\n`;
    summary += indent + `   p95 Duration: ${(data.metrics.basic_request_duration.values['p(95)'] || 0).toFixed(0)}ms\n`;
    summary += indent + `   Max Duration: ${(data.metrics.basic_request_duration.values.max || 0).toFixed(0)}ms\n\n`;
  }

  // Training Insights
  summary += indent + '🧠 RL TRAINING INSIGHTS\n';
  summary += indent + '   Load Phases Tested:\n';
  summary += indent + '   ✓ Night-time (minimal load)\n';
  summary += indent + '   ✓ Morning ramp-up (gradual increase)\n';
  summary += indent + '   ✓ Steady daytime (stable load)\n';
  summary += indent + '   ✓ Lunch dip (temporary decrease)\n';
  summary += indent + '   ✓ Afternoon peak (sustained high load)\n';
  summary += indent + '   ✓ Flash spike (sudden burst)\n';
  summary += indent + '   ✓ Evening decline (gradual decrease)\n';
  summary += indent + '   ✓ Oscillating patterns (rapid changes)\n\n';

  summary += indent + '   Scenarios Covered:\n';
  summary += indent + '   ✓ Gradual scaling up and down\n';
  summary += indent + '   ✓ Sustained high load endurance\n';
  summary += indent + '   ✓ Sudden traffic spikes\n';
  summary += indent + '   ✓ Rapid load fluctuations\n';
  summary += indent + '   ✓ Mixed CPU/Memory/Basic workloads\n';
  summary += indent + '   ✓ Realistic daily traffic patterns\n\n';

  // Recommendations for RL Agent
  summary += indent + '💡 NEXT STEPS FOR RL AGENT\n';
  summary += indent + '   1. Review metrics in InfluxDB/Prometheus\n';
  summary += indent + '   2. Analyze scaling decisions vs. load phases\n';
  summary += indent + '   3. Check reward patterns for each scenario\n';
  summary += indent + '   4. Verify cost optimization during low load\n';
  summary += indent + '   5. Validate response time SLA compliance\n';
  summary += indent + '   6. Assess scale-down behavior post-spike\n\n';

  // Threshold Compliance
  const thresholdsPassed = [];
  const thresholdsFailed = [];

  if (data.metrics.http_req_duration?.values['p(95)'] < 8000) {
    thresholdsPassed.push('p95 response time < 8s');
  } else {
    thresholdsFailed.push('p95 response time < 8s');
  }

  if (failedRate < 0.12) {
    thresholdsPassed.push('Error rate < 12%');
  } else {
    thresholdsFailed.push('Error rate < 12%');
  }

  summary += indent + '✅ THRESHOLD COMPLIANCE\n';
  if (thresholdsPassed.length > 0) {
    summary += indent + '   Passed:\n';
    thresholdsPassed.forEach(t => summary += indent + `   ✓ ${t}\n`);
  }
  if (thresholdsFailed.length > 0) {
    summary += indent + '   Failed:\n';
    thresholdsFailed.forEach(t => summary += indent + `   ✗ ${t}\n`);
  }

  summary += indent + '\n═══════════════════════════════════════════════════════\n';
  summary += indent + '🎓 Training data has been collected for RL agent optimization\n';
  summary += indent + '═══════════════════════════════════════════════════════\n\n';

  return summary;
}
