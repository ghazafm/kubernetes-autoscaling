import { check, sleep } from 'k6';
import exec from 'k6/execution';
import http from 'k6/http';
import { Counter, Gauge, Rate, Trend } from 'k6/metrics';


// Custom metrics for comprehensive monitoring
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration', true);
const memoryDuration = new Trend('memory_request_duration', true);
const basicDuration = new Trend('basic_request_duration', true);
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

// Dynamic load calculation based on replica capacity
// This ensures k6 generates enough load to stress the full replica range
const MAX_REPLICAS = parseInt(__ENV.MAX_REPLICAS || '50');
const MIN_REPLICAS = parseInt(__ENV.MIN_REPLICAS || '1');
const REQUESTS_PER_POD_TARGET = parseFloat(__ENV.REQUESTS_PER_POD || '8');

// Calculate VU targets to stress pods at different capacity levels
// Formula: VUs = replicas * requests_per_pod * utilization_target
const VU_WARMUP = Math.ceil(MIN_REPLICAS * 2);  // Minimal load
const VU_LOW = Math.ceil(MAX_REPLICAS * 0.2 * REQUESTS_PER_POD_TARGET);  // 20% capacity
const VU_MEDIUM = Math.ceil(MAX_REPLICAS * 0.4 * REQUESTS_PER_POD_TARGET);  // 40% capacity
const VU_HIGH = Math.ceil(MAX_REPLICAS * 0.6 * REQUESTS_PER_POD_TARGET);  // 60% capacity
const VU_PEAK = Math.ceil(MAX_REPLICAS * 0.8 * REQUESTS_PER_POD_TARGET);  // 80% capacity
const VU_SPIKE = Math.ceil(MAX_REPLICAS * 1.0 * REQUESTS_PER_POD_TARGET);  // 100% capacity

// Helper to ensure integer VU targets (k6 requires integers)
function ensureInt(value) {
  return Math.ceil(value);
}

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
// Now uses dynamic VU targets based on MAX_REPLICAS
const basePattern = [
  // ===== WARM-UP PHASE (Baseline establishment) =====
  { duration: scaleDuration(1), target: 0 },               // No requests initially
  { duration: scaleDuration(1), target: VU_WARMUP },       // Gentle start
  { duration: scaleDuration(2), target: VU_WARMUP },       // Baseline LOW traffic

  // ===== PHASE 1: GRADUAL MORNING RAMP-UP (Simulates business hours start) =====
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.5) },    // Early morning users arrive
  { duration: scaleDuration(2), target: ensureInt(VU_LOW * 0.5) },    // Low morning traffic
  { duration: scaleDuration(1), target: VU_LOW },          // More users logging in
  { duration: scaleDuration(2), target: VU_LOW },          // Growing morning traffic
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 1.5) },    // Peak morning traffic
  { duration: scaleDuration(3), target: ensureInt(VU_LOW * 1.5) },    // Sustained morning activity

  // ===== PHASE 2: STEADY DAYTIME LOAD (Normal business operations) =====
  { duration: scaleDuration(1), target: VU_MEDIUM },       // Midday increase
  { duration: scaleDuration(4), target: VU_MEDIUM },       // Stable daytime load

  // ===== PHASE 3: LUNCH HOUR DIP (Realistic traffic pattern) =====
  { duration: scaleDuration(1), target: VU_LOW },          // Users taking lunch break
  { duration: scaleDuration(2), target: VU_LOW },          // Reduced lunch activity

  // ===== PHASE 4: POST-LUNCH RECOVERY =====
  { duration: scaleDuration(1), target: ensureInt(VU_MEDIUM * 1.2) }, // Users returning
  { duration: scaleDuration(3), target: ensureInt(VU_MEDIUM * 1.2) }, // Afternoon steady state

  // ===== PHASE 5: AFTERNOON PEAK (Highest daily load) =====
  { duration: scaleDuration(1), target: VU_HIGH },         // Building to peak
  { duration: scaleDuration(2), target: VU_PEAK },         // Ramp to peak
  { duration: scaleDuration(4), target: VU_PEAK },         // Sustained peak load

  // ===== PHASE 6: FLASH SPIKE (Sudden event - viral content, promotion) =====
  { duration: scaleDuration(0.5), target: VU_SPIKE },      // Sudden viral spike
  { duration: scaleDuration(2), target: VU_SPIKE },        // Spike sustained
  { duration: scaleDuration(1), target: VU_PEAK },         // Quick recovery to normal peak

  // ===== PHASE 7: GRADUAL EVENING DECLINE =====
  { duration: scaleDuration(1), target: VU_HIGH },         // Early evening decrease
  { duration: scaleDuration(2), target: VU_MEDIUM },       // Continued decline
  { duration: scaleDuration(2), target: ensureInt(VU_LOW * 1.5) },    // Further decrease
  { duration: scaleDuration(2), target: VU_LOW },          // Late evening

  // ===== PHASE 8: NIGHT-TIME LOW (Maintenance window simulation) =====
  { duration: scaleDuration(1), target: ensureInt(VU_WARMUP * 2) },   // Night users
  { duration: scaleDuration(3), target: ensureInt(VU_WARMUP * 2) },   // Sustained low load
  { duration: scaleDuration(1), target: VU_WARMUP },       // Deep night
  { duration: scaleDuration(2), target: VU_WARMUP },       // Minimal activity

  // ===== PHASE 9: OSCILLATING LOAD (Test rapid adaptation) =====
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 1.5) },  // Quick up
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 1.5) },    // Hold
  { duration: scaleDuration(0.5), target: ensureInt(VU_WARMUP * 3) }, // Quick down
  { duration: scaleDuration(1), target: ensureInt(VU_WARMUP * 3) },   // Hold
  { duration: scaleDuration(0.5), target: VU_MEDIUM },     // Quick up again
  { duration: scaleDuration(1), target: VU_MEDIUM },       // Hold
  { duration: scaleDuration(0.5), target: ensureInt(VU_WARMUP * 2) }, // Quick down

  // ===== PHASE 10: GRACEFUL SHUTDOWN =====
  { duration: scaleDuration(1), target: VU_WARMUP },       // Final users
  { duration: scaleDuration(2), target: 0 },             // Complete ramp down
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

// Support multiple target URLs via BASE_URLS (comma-separated) or single BASE_URL
const BASE_URLS_RAW = __ENV.BASE_URLS || __ENV.BASE_URL || 'http://localhost:5000';
const BASE_URLS = BASE_URLS_RAW.split(',').map(s => s.trim()).filter(Boolean);

// Helper to pick a base URL. Uses __VU mapping for stable distribution across VUs,
// falls back to a random choice when __VU is not available.
function getBaseUrl() {
  if (BASE_URLS.length === 1) return BASE_URLS[0];
  const vuId = exec.vu.idInTest;
  return BASE_URLS[(vuId - 1) % BASE_URLS.length];
}

const MAX_CPU_ITERATIONS = parseInt(__ENV.MAX_CPU_ITERATIONS || '500000');

function safeGet(url, params, maxRetries = 2) {
  let attempt = 0;
  while (attempt <= maxRetries) {
    try {
      const r = http.get(url, params);
      return r;
    } catch (err) {
      attempt += 1;
      sleep(0.1 * attempt);
      if (attempt > maxRetries) throw err;
    }
  }
}

// Request type distribution that mimics real-world application usage
const REQUEST_PATTERNS = {
  LIGHT: { cpu: 0.10, memory: 0.05, basic: 0.85 },      // Night-time pattern
  NORMAL: { cpu: 0.30, memory: 0.20, basic: 0.50 },     // Regular business hours
  INTENSIVE: { cpu: 0.45, memory: 0.35, basic: 0.20 },  // Peak hours pattern
  SPIKE: { cpu: 0.40, memory: 0.40, basic: 0.20 },      // Viral/spike pattern
};

// Determine current load phase based on VU count
function getLoadPhase(activeVUs) {
  if (activeVUs <= 3) return 'NIGHT';
  if (activeVUs <= 10) return 'LOW';
  if (activeVUs <= 20) return 'MEDIUM';
  if (activeVUs <= 30) return 'HIGH';
  if (activeVUs <= 50) return 'PEAK';
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
// REDUCED to avoid long-tail latency and request queueing under high concurrency
function getCpuIterations(phase) {
  let base, variance;

  switch(phase) {
    case 'NIGHT':
    case 'LOW':
      base = 200000;      // Light CPU tasks (200k-500k iterations)
      variance = 300000;  // ~0.5-2 seconds CPU time
      break;
    case 'MEDIUM':
      base = 400000;      // Moderate CPU tasks (400k-800k iterations)
      variance = 400000;  // ~1-3 seconds CPU time
      break;
    case 'HIGH':
      base = 600000;      // Heavy CPU tasks (600k-1M iterations)
      variance = 400000;  // ~2-4 seconds CPU time
      break;
    case 'PEAK':
    case 'EXTREME':
      base = 800000;      // Maximum CPU tasks (800k-1.2M iterations)
      variance = 400000;  // ~3-5 seconds CPU time (avoids queueing at high VU)
      break;
    default:
      base = 500000;
      variance = 300000;
  }

  return Math.floor(base + Math.random() * variance);
}

// Generate memory size based on load phase and realism
// Optimized for 512Mi container limit with MAX_MEMORY_MB=140
// CRITICAL: Max request must account for concurrency (2-3 simultaneous requests)
// Formula: Safe_Max = MAX_MEMORY_MB ÷ 2 = 140 ÷ 2 = 70 MB
function getMemorySize(phase) {
  let base, variance;

  switch(phase) {
    case 'NIGHT':
    case 'LOW':
      base = 10;          // Minimal allocations (10-20 MB)
      variance = 10;
      break;
    case 'MEDIUM':
      base = 25;          // Light allocations (25-40 MB)
      variance = 15;
      break;
    case 'HIGH':
      base = 40;          // Moderate allocations (40-55 MB)
      variance = 15;
      break;
    case 'PEAK':
    case 'EXTREME':
      base = 50;          // Maximum allocations (50-70 MB)
      variance = 20;      // Up to 70 MB max (safe for 2-3 concurrent requests)
      break;
    default:
      base = 30;
      variance = 15;
  }

  return Math.floor(base + Math.random() * variance);
}

export function setup() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('🎯 RL AUTOSCALER TRAINING - DYNAMIC LOAD');
  console.log('═══════════════════════════════════════════════════════');
  if (BASE_URLS.length === 1) {
    console.log(`Target: ${BASE_URLS[0]}`);
  } else {
    console.log(`Targets: ${BASE_URLS.join(', ')}`);
  }
  console.log(`Duration Multiplier: ${DURATION_MULTIPLIER}x`);
  console.log(`Cycle Count: ${CYCLE_COUNT}`);
  console.log('');
  console.log('📊 Replica Configuration:');
  console.log(`   MIN_REPLICAS: ${MIN_REPLICAS}`);
  console.log(`   MAX_REPLICAS: ${MAX_REPLICAS}`);
  console.log(`   Target Requests/Pod: ${REQUESTS_PER_POD_TARGET}`);
  console.log('');
  console.log('🚀 Dynamic VU Targets:');
  console.log(`   WARMUP: ${VU_WARMUP} VUs (${MIN_REPLICAS}-${Math.ceil(VU_WARMUP/REQUESTS_PER_POD_TARGET)} pods)`);
  console.log(`   LOW:    ${VU_LOW} VUs (~${Math.ceil(VU_LOW/REQUESTS_PER_POD_TARGET)} pods at 80% util)`);
  console.log(`   MEDIUM: ${VU_MEDIUM} VUs (~${Math.ceil(VU_MEDIUM/REQUESTS_PER_POD_TARGET)} pods at 80% util)`);
  console.log(`   HIGH:   ${VU_HIGH} VUs (~${Math.ceil(VU_HIGH/REQUESTS_PER_POD_TARGET)} pods at 80% util)`);
  console.log(`   PEAK:   ${VU_PEAK} VUs (~${Math.ceil(VU_PEAK/REQUESTS_PER_POD_TARGET)} pods at 80% util)`);
  console.log(`   SPIKE:  ${VU_SPIKE} VUs (~${Math.ceil(VU_SPIKE/REQUESTS_PER_POD_TARGET)} pods at 80% util)`);
  console.log('');
  console.log('💡 RL Agent Training Coverage:');
  console.log(`   This load will train across ${MIN_REPLICAS}-${MAX_REPLICAS} replica range`);
  console.log(`   Each pod will experience 60-100% utilization at PEAK`);
  console.log('═══════════════════════════════════════════════════════\n');
}

export default function () {
  const activeVUs = exec.instance.vusActive;   // real current load
  const phase = getLoadPhase(activeVUs);
  const pattern = getRequestPattern(phase);

  currentLoad.add(activeVUs);

  // Update gauge metric for monitoring
  currentLoad.add(activeVUs);

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
      const iterations = Math.min(getCpuIterations(phase), MAX_CPU_ITERATIONS);
      res = safeGet(`${getBaseUrl()}/api/cpu?iterations=${iterations}`, {
        tags: { name: 'cpu', request_type: 'cpu', load_phase: phase },
        timeout: '30s', // Prevent hanging requests
      });

      cpuDuration.add(res.timings.duration);

      const cpuOk = check(res, {
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
      });
      errorRate.add(!cpuOk);;
      break;

    case 'memory':
      const sizeMb = getMemorySize(phase);
      res = safeGet(`${getBaseUrl()}/api/memory?size_mb=${sizeMb}`, {
        tags: { name: 'memory', request_type: 'memory', load_phase: phase },
        timeout: '20s',
      });

      memoryDuration.add(res.timings.duration);

      const memOk = check(res, {
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
      });
      errorRate.add(memOk ? 0 : 1);
      break;

    case 'basic':
      res = safeGet(`${getBaseUrl()}/api`, {
        tags: { name: 'basic', request_type: 'basic', load_phase: phase },
        timeout: '5s',
      });

      basicDuration.add(res.timings.duration);

      const basicOk = check(res, {
        'basic api status is 200': (r) => r.status === 200,
        'basic api response time acceptable': (r) => r.timings.duration < 2000,
      });
      errorRate.add(basicOk ? 0 : 1);
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
