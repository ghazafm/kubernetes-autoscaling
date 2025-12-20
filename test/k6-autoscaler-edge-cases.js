import { check, sleep } from 'k6';
import exec from 'k6/execution';
import http from 'k6/http';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');
const memoryDuration = new Trend('memory_request_duration');
const edgeCaseCounter = new Counter('edge_case_counter');

// RL Autoscaler Edge Cases Test
// This test focuses on challenging scenarios that help the RL agent learn robust policies
//
// CUSTOMIZABLE DURATION:
// Set DURATION_MULTIPLIER environment variable to extend test duration
// Set CYCLE_COUNT to repeat the pattern multiple times
//
// Usage:
//   k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=7 k6-autoscaler-edge-cases.js

const DURATION_MULTIPLIER = parseFloat(__ENV.DURATION_MULTIPLIER || '1');
const CYCLE_COUNT = parseInt(__ENV.CYCLE_COUNT || '1');
// Allow k6 to respect the app's configured CPU cap if provided
const MAX_CPU_ITERATIONS = parseInt(__ENV.MAX_CPU_ITERATIONS || '500000');

// Dynamic load calculation based on replica capacity
// This ensures k6 generates enough load to stress the full replica range
const MAX_REPLICAS = parseInt(__ENV.MAX_REPLICAS || '50');
const MIN_REPLICAS = parseInt(__ENV.MIN_REPLICAS || '1');
const REQUESTS_PER_POD_TARGET = parseFloat(__ENV.REQUESTS_PER_POD || '8');

// Calculate VU targets to stress pods at different capacity levels
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

// Base pattern (40 minutes cycle)
// Now uses dynamic VU targets based on MAX_REPLICAS
const basePattern = [
  // ===== SCENARIO 1: COLD START (from 0 to moderate load) =====
  { duration: scaleDuration(0.5), target: 0 },                         // Ensure cold start
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.5) },   // Rapid cold start ramp
  { duration: scaleDuration(2), target: ensureInt(VU_LOW * 0.5) },     // Sustain to test stability

  // ===== SCENARIO 2: THUNDERING HERD (extreme sudden spike) =====
  { duration: scaleDuration(0.33), target: VU_WARMUP },                // Very low baseline
  { duration: scaleDuration(0.33), target: VU_SPIKE },                 // Massive instant spike
  { duration: scaleDuration(2), target: VU_SPIKE },                    // Sustain extreme load
  { duration: scaleDuration(0.5), target: VU_WARMUP },                 // Rapid drop (test scale-down)

  // ===== SCENARIO 3: SAWTOOTH PATTERN (repeated spikes) =====
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.5) },   // Baseline
  { duration: scaleDuration(0.5), target: VU_MEDIUM },                 // Spike 1
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.5) },   // Drop
  { duration: scaleDuration(0.5), target: VU_MEDIUM },                 // Spike 2
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.5) },   // Drop
  { duration: scaleDuration(0.5), target: VU_MEDIUM },                 // Spike 3
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.5) },   // Drop

  // ===== SCENARIO 4: SLOW LEAK (gradual sustained increase) =====
  { duration: scaleDuration(1), target: VU_WARMUP },                   // Start low
  { duration: scaleDuration(5), target: VU_HIGH },                     // Very gradual increase
  { duration: scaleDuration(2), target: VU_HIGH },                     // Hold at high

  // ===== SCENARIO 5: STAIRCASE PATTERN (discrete load levels) =====
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.5) },     // Step 1
  { duration: scaleDuration(1), target: VU_LOW },                      // Step 2
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 1.5) },     // Step 3
  { duration: scaleDuration(1), target: VU_MEDIUM },                   // Step 4
  { duration: scaleDuration(1), target: VU_HIGH },                     // Step 5 (peak)
  { duration: scaleDuration(1), target: VU_MEDIUM },                   // Step down
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 1.5) },     // Step down
  { duration: scaleDuration(1), target: VU_LOW },                      // Step down
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.5) },     // Step down

  // ===== SCENARIO 6: JITTER PATTERN (noisy load) =====
  { duration: scaleDuration(0.33), target: ensureInt(VU_LOW * 0.75) }, // Base
  { duration: scaleDuration(0.33), target: VU_LOW },                   // Jitter up
  { duration: scaleDuration(0.33), target: ensureInt(VU_LOW * 0.6) },  // Jitter down
  { duration: scaleDuration(0.33), target: ensureInt(VU_LOW * 0.9) },  // Jitter up
  { duration: scaleDuration(0.33), target: ensureInt(VU_LOW * 0.7) },  // Jitter down
  { duration: scaleDuration(0.33), target: ensureInt(VU_LOW * 1.1) },  // Jitter up
  { duration: scaleDuration(0.33), target: ensureInt(VU_LOW * 0.8) },  // Jitter down

  // ===== SCENARIO 7: SUSTAINED MAXIMUM (endurance test) =====
  { duration: scaleDuration(1), target: VU_PEAK },                     // Ramp to maximum
  { duration: scaleDuration(4), target: VU_PEAK },                     // Sustain maximum load

  // ===== SCENARIO 8: RAPID OSCILLATION (high frequency changes) =====
  { duration: scaleDuration(0.33), target: ensureInt(VU_LOW * 0.5) },  // Low
  { duration: scaleDuration(0.33), target: ensureInt(VU_MEDIUM * 0.75) }, // High
  { duration: scaleDuration(0.33), target: ensureInt(VU_LOW * 0.5) },  // Low
  { duration: scaleDuration(0.33), target: ensureInt(VU_MEDIUM * 0.75) }, // High
  { duration: scaleDuration(0.33), target: ensureInt(VU_LOW * 0.5) },  // Low
  { duration: scaleDuration(0.33), target: ensureInt(VU_MEDIUM * 0.75) }, // High

  // ===== SCENARIO 9: ASYMMETRIC RAMP (slow up, fast down) =====
  { duration: scaleDuration(3), target: ensureInt(VU_MEDIUM * 0.9) },  // Slow ramp up
  { duration: scaleDuration(0.5), target: VU_WARMUP },                 // Rapid drop
  { duration: scaleDuration(1), target: VU_WARMUP },                   // Hold low

  // ===== SCENARIO 10: DEAD ZONE (minimal load) =====
  { duration: scaleDuration(0.5), target: 1 },                         // Near zero
  { duration: scaleDuration(2), target: 1 },                           // Sustain near zero

  // ===== FINAL: GRACEFUL SHUTDOWN =====
  { duration: scaleDuration(0.5), target: 0 },                         // Complete shutdown
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
    http_req_duration: ['p(95)<15000'], // Allow more time for edge cases
    errors: ['rate<0.25'],               // Higher tolerance for edge scenarios
  },
};

// Support multiple target URLs via BASE_URLS (comma-separated) or single BASE_URL
const BASE_URLS_RAW = __ENV.BASE_URLS || __ENV.BASE_URL || 'http://localhost:5000';
const BASE_URLS = BASE_URLS_RAW.split(',').map(s => s.trim()).filter(Boolean);

function getBaseUrl() {
  if (BASE_URLS.length === 1) return BASE_URLS[0];
  if (typeof exec !== 'undefined' && exec.vu && exec.vu.idInTest) {
    const vuId = exec.vu.idInTest;
    return BASE_URLS[(vuId - 1) % BASE_URLS.length];
  }
  if (typeof __VU !== 'undefined') return BASE_URLS[(__VU - 1) % BASE_URLS.length];
  return BASE_URLS[Math.floor(Math.random() * BASE_URLS.length)];
}

export function setup() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('🎯 RL AUTOSCALER EDGE CASES - DYNAMIC LOAD');
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
  console.log(`   WARMUP: ${VU_WARMUP} VUs`);
  console.log(`   LOW:    ${VU_LOW} VUs`);
  console.log(`   MEDIUM: ${VU_MEDIUM} VUs`);
  console.log(`   HIGH:   ${VU_HIGH} VUs`);
  console.log(`   PEAK:   ${VU_PEAK} VUs`);
  console.log(`   SPIKE:  ${VU_SPIKE} VUs`);
  console.log('═══════════════════════════════════════════════════════\n');
}

// Determine edge case scenario based on time and VU (dynamic thresholds based on MAX_REPLICAS)
function getEdgeCaseScenario(vu) {
  if (vu === 0 || vu === 1) return 'DEAD_ZONE';
  if (vu >= VU_PEAK) return 'EXTREME_LOAD';
  if (vu >= VU_HIGH) return 'HIGH_PRESSURE';
  if (vu <= VU_WARMUP) return 'MINIMAL_LOAD';

  // Check for rapid changes (oscillation patterns)
  const oscillationCheck = Math.floor(__ITER / 10) % 2;
  if (vu >= VU_MEDIUM && oscillationCheck === 0) return 'OSCILLATING_HIGH';
  if (vu <= VU_LOW && oscillationCheck === 1) return 'OSCILLATING_LOW';

  return 'MODERATE_EDGE';
}

// Generate extreme workloads for edge cases
function getExtremeWorkload(scenario) {
  switch(scenario) {
    case 'DEAD_ZONE':
    case 'MINIMAL_LOAD':
      return {
        type: 'basic',
        intensity: 'minimal',
      };

    case 'EXTREME_LOAD':
      return {
        type: Math.random() < 0.6 ? 'cpu' : 'memory',
        intensity: 'maximum',
      };

    case 'HIGH_PRESSURE':
      return {
        type: Math.random() < 0.7 ? 'cpu' : 'memory',
        intensity: 'high',
      };

    case 'OSCILLATING_HIGH':
      return {
        type: Math.random() < 0.5 ? 'cpu' : 'memory',
        intensity: 'burst',
      };

    case 'OSCILLATING_LOW':
      return {
        type: Math.random() < 0.7 ? 'basic' : 'cpu',
        intensity: 'light',
      };

    default:
      return {
        type: Math.random() < 0.4 ? 'cpu' : (Math.random() < 0.5 ? 'memory' : 'basic'),
        intensity: 'moderate',
      };
  }
}

// Get workload parameters based on intensity
function getWorkloadParams(type, intensity) {
  let params;

  if (type === 'cpu') {
    // All ranges capped to MAX_CPU_ITERATIONS (500000)
    switch(intensity) {
      case 'minimal':
        params = { iterations: 50000 + Math.floor(Math.random() * 50000) };  // 50k-100k
        break;
      case 'light':
        params = { iterations: 100000 + Math.floor(Math.random() * 100000) }; // 100k-200k
        break;
      case 'moderate':
        params = { iterations: 200000 + Math.floor(Math.random() * 150000) }; // 200k-350k
        break;
      case 'high':
        params = { iterations: 300000 + Math.floor(Math.random() * 150000) }; // 300k-450k
        break;
      case 'burst':
        params = { iterations: 350000 + Math.floor(Math.random() * 100000) }; // 350k-450k
        break;
      case 'maximum':
        // Cap to MAX_CPU_ITERATIONS (provided via environment/config)
        params = { iterations: Math.min(400000 + Math.floor(Math.random() * 100000), MAX_CPU_ITERATIONS) }; // 400k-500k
        break;
      default:
        params = { iterations: 250000 };
    }
  } else if (type === 'memory') {
    // FIXED: Reduced all values to max 70 MB for concurrency safety
    switch(intensity) {
      case 'minimal':
        params = { size_mb: 10 + Math.floor(Math.random() * 10) }; // 10-20 MB
        break;
      case 'light':
        params = { size_mb: 20 + Math.floor(Math.random() * 15) }; // 20-35 MB
        break;
      case 'moderate':
        params = { size_mb: 30 + Math.floor(Math.random() * 15) }; // 30-45 MB
        break;
      case 'high':
        params = { size_mb: 40 + Math.floor(Math.random() * 15) }; // 40-55 MB
        break;
      case 'burst':
        params = { size_mb: 45 + Math.floor(Math.random() * 15) }; // 45-60 MB
        break;
      case 'maximum':
        params = { size_mb: 50 + Math.floor(Math.random() * 20) }; // 50-70 MB (safe max)
        break;
      default:
        params = { size_mb: 35 };
    }
  }

  return params;
}

export default function () {
  const scenario = getEdgeCaseScenario(__VU);
  const workload = getExtremeWorkload(scenario);

  // Track edge case occurrence
  edgeCaseCounter.add(1, { scenario: scenario });

  let res;

  // Helper to retry transient network errors (connect refused / timeouts)
  function safeGet(url, params, maxRetries = 2) {
    let attempt = 0;
    while (attempt <= maxRetries) {
      try {
        const r = http.get(url, params);
        return r;
      } catch (err) {
        attempt += 1;
        // small backoff
        sleep(0.1 * attempt);
        if (attempt > maxRetries) {
          // rethrow to allow the caller to record the error
          throw err;
        }
      }
    }
  }

  if (workload.type === 'cpu') {
    const params = getWorkloadParams('cpu', workload.intensity);
    // Use a stable name tag to avoid URL-based high-cardinality in k6 metrics
    res = safeGet(`${getBaseUrl()}/api/cpu?iterations=${params.iterations}`, {
      tags: {
        name: 'cpu',
        request_type: 'cpu',
        scenario: scenario,
        intensity: workload.intensity,
      },
      timeout: '40s',
    });

    cpuDuration.add(res.timings.duration);

    check(res, {
      'cpu edge case status is 200': (r) => r.status === 200,
      'cpu edge case completed': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'success';
        } catch (e) {
          return false;
        }
      },
    }) || errorRate.add(1);

  } else if (workload.type === 'memory') {
    const params = getWorkloadParams('memory', workload.intensity);
    // Add a stable "name" tag to avoid high-cardinality series caused by
    // many distinct query parameter combinations being used as metric tags.
    res = safeGet(`${getBaseUrl()}/api/memory?size_mb=${params.size_mb}`, {
      tags: {
        name: 'memory',
        request_type: 'memory',
        scenario: scenario,
        intensity: workload.intensity,
      },
      timeout: '30s',
    });

    memoryDuration.add(res.timings.duration);

    check(res, {
      'memory edge case status is 200': (r) => r.status === 200,
      'memory edge case completed': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'success';
        } catch (e) {
          return false;
        }
      },
    }) || errorRate.add(1);

  } else {
    // Basic request
    res = safeGet(`${getBaseUrl()}/api`, {
      tags: {
        name: 'basic',
        request_type: 'basic',
        scenario: scenario,
        intensity: workload.intensity,
      },
      timeout: '10s',
    });

    check(res, {
      'basic edge case status is 200': (r) => r.status === 200,
    }) || errorRate.add(1);
  }

  // Adaptive sleep based on scenario
  let sleepTime;
  switch(scenario) {
    case 'DEAD_ZONE':
    case 'MINIMAL_LOAD':
      sleepTime = 3 + Math.random() * 2; // 3-5 seconds
      break;
    case 'EXTREME_LOAD':
      sleepTime = 0.1 + Math.random() * 0.2; // 0.1-0.3 seconds (very high pressure)
      break;
    case 'HIGH_PRESSURE':
      sleepTime = 0.3 + Math.random() * 0.3; // 0.3-0.6 seconds
      break;
    case 'OSCILLATING_HIGH':
      sleepTime = 0.2 + Math.random() * 0.3; // 0.2-0.5 seconds
      break;
    case 'OSCILLATING_LOW':
      sleepTime = 1.5 + Math.random() * 1.0; // 1.5-2.5 seconds
      break;
    default:
      sleepTime = 0.8 + Math.random() * 0.8; // 0.8-1.6 seconds
  }

  sleep(sleepTime);
}

export function handleSummary(data) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
  const timeHour = new Date().toISOString().replace(/[:.]/g, '-').split('T')[1].substring(0, 5);

  return {
    [`edge-cases-summary-${timestamp}-${timeHour}.json`]: JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options = {}) {
  const indent = options.indent || '';

  let summary = '\n' + indent + '🎯 RL AUTOSCALER EDGE CASES TEST - RESULTS\n';
  summary += indent + '═══════════════════════════════════════════════════════\n\n';

  // Test Configuration
  summary += indent + '⚙️  TEST CONFIGURATION\n';
  summary += indent + `   Duration Multiplier: ${DURATION_MULTIPLIER}x\n`;
  summary += indent + `   Cycle Count: ${CYCLE_COUNT}\n`;
  summary += indent + `   Base Pattern: 40 minutes → Actual: ${(40 * DURATION_MULTIPLIER).toFixed(1)} min per cycle\n`;
  summary += indent + `   Total Planned Duration: ${(40 * DURATION_MULTIPLIER * CYCLE_COUNT / 60).toFixed(1)} hours (${(40 * DURATION_MULTIPLIER * CYCLE_COUNT / 60 / 24).toFixed(1)} days)\n\n`;

  const duration = (data.state?.testRunDurationMs / 1000 / 60) || 0;
  const totalRequests = data.metrics.http_reqs?.values.count || 0;
  const failedRate = data.metrics.http_req_failed?.values.rate || 0;
  const successRate = (1 - failedRate) * 100;

  summary += indent + '⏱️  TEST OVERVIEW\n';
  summary += indent + `   Duration: ${duration.toFixed(1)} minutes (${(duration/60).toFixed(1)} hours / ${(duration/60/24).toFixed(2)} days)\n`;
  summary += indent + `   Total Requests: ${totalRequests.toLocaleString()}\n`;
  summary += indent + `   Success Rate: ${successRate.toFixed(2)}%\n`;
  summary += indent + `   Error Rate: ${(failedRate * 100).toFixed(2)}%\n\n`;

  summary += indent + '📊 RESPONSE TIME METRICS\n';
  summary += indent + `   Average: ${(data.metrics.http_req_duration?.values.avg || 0).toFixed(0)}ms\n`;
  summary += indent + `   Median: ${(data.metrics.http_req_duration?.values.med || 0).toFixed(0)}ms\n`;
  summary += indent + `   p95: ${(data.metrics.http_req_duration?.values['p(95)'] || 0).toFixed(0)}ms\n`;
  summary += indent + `   p99: ${(data.metrics.http_req_duration?.values['p(99)'] || 0).toFixed(0)}ms\n`;
  summary += indent + `   Max: ${(data.metrics.http_req_duration?.values.max || 0).toFixed(0)}ms\n\n`;

  summary += indent + '🔬 EDGE CASES TESTED\n';
  summary += indent + '   ✓ Cold Start (0 → moderate load)\n';
  summary += indent + '   ✓ Thundering Herd (extreme sudden spike)\n';
  summary += indent + '   ✓ Sawtooth Pattern (repeated spikes)\n';
  summary += indent + '   ✓ Slow Leak (gradual sustained increase)\n';
  summary += indent + '   ✓ Staircase Pattern (discrete levels)\n';
  summary += indent + '   ✓ Jitter Pattern (noisy load)\n';
  summary += indent + '   ✓ Sustained Maximum (endurance)\n';
  summary += indent + '   ✓ Rapid Oscillation (high frequency)\n';
  summary += indent + '   ✓ Asymmetric Ramp (slow up, fast down)\n';
  summary += indent + '   ✓ Dead Zone (minimal load)\n\n';

  summary += indent + '💡 RL AGENT LEARNING OPPORTUNITIES\n';
  summary += indent + '   • Handling sudden load spikes efficiently\n';
  summary += indent + '   • Aggressive scale-down without SLA violation\n';
  summary += indent + '   • Stability during oscillating loads\n';
  summary += indent + '   • Resource optimization at extreme loads\n';
  summary += indent + '   • Recovery from near-zero states\n';
  summary += indent + '   • Predictive scaling for gradual changes\n\n';

  summary += indent + '═══════════════════════════════════════════════════════\n';
  summary += indent + '🧠 Edge case training data collected successfully\n';
  summary += indent + '═══════════════════════════════════════════════════════\n\n';

  return summary;
}
