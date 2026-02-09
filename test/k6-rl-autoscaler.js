import { check, sleep } from 'k6';
import http from 'k6/http';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');
const memoryDuration = new Trend('memory_request_duration');
const requestsPerStage = new Counter('requests_per_stage');

// Support multiple target URLs via BASE_URLS (comma-separated) or single BASE_URL
const BASE_URLS_RAW = __ENV.BASE_URLS || __ENV.BASE_URL || 'http://localhost:5000';
const BASE_URLS = BASE_URLS_RAW.split(',').map(s => s.trim()).filter(Boolean);
const URL_COUNT = BASE_URLS.length;

// Dynamic load calculation based on replica capacity
const MAX_REPLICAS = parseInt(__ENV.MAX_REPLICAS || '50');
const MIN_REPLICAS = parseInt(__ENV.MIN_REPLICAS || '1');
const REQUESTS_PER_POD_TARGET = parseFloat(__ENV.REQUESTS_PER_POD || '8');

// Calculate VU targets to stress pods at different capacity levels
// Formula: VUs = replicas * requests_per_pod * url_count * utilization_target
const VU_LOW = Math.max(1, Math.ceil(MAX_REPLICAS * 0.2 * REQUESTS_PER_POD_TARGET * URL_COUNT));
const VU_MEDIUM = Math.ceil(MAX_REPLICAS * 0.4 * REQUESTS_PER_POD_TARGET * URL_COUNT);
const VU_HIGH = Math.ceil(MAX_REPLICAS * 0.6 * REQUESTS_PER_POD_TARGET * URL_COUNT);
const VU_PEAK = Math.ceil(MAX_REPLICAS * 0.8 * REQUESTS_PER_POD_TARGET * URL_COUNT);
const VU_SPIKE = Math.ceil(MAX_REPLICAS * 1.0 * REQUESTS_PER_POD_TARGET * URL_COUNT);

// Cap all VU calculations at VU_SPIKE to never exceed pod capacity
const vu = (v) => Math.min(Math.max(1, Math.ceil(v)), VU_SPIKE);

// RL Autoscaler Test Configuration - Dynamic VU based on MAX_REPLICAS
// Scenario: Low → Medium → High → Low (to test RL agent's scaling decisions)
export const options = {
  stages: [
    // Start
    { duration: '1m', target: 1 },

    // Phase 1: LOW LOAD (baseline)
    { duration: '2m', target: vu(VU_LOW * 0.5) },  // Ramp up slowly
    { duration: '3m', target: vu(VU_LOW * 0.5) },  // Hold at half low

    // Phase 2: MEDIUM LOAD
    { duration: '1m', target: VU_LOW },         // Ramp up to low
    { duration: '4m', target: VU_LOW },         // Hold at low (MEDIUM)

    // Phase 3: HIGH LOAD
    { duration: '1m', target: VU_MEDIUM },      // Ramp up to medium
    { duration: '5m', target: VU_MEDIUM },      // Hold at medium (HIGH)

    // Phase 4: BACK TO LOW (scale down test)
    { duration: '1m', target: vu(VU_LOW * 0.5) },  // Ramp down
    { duration: '3m', target: vu(VU_LOW * 0.5) },  // Hold at half low

    // Graceful shutdown
    { duration: '30s', target: 0 },             // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests should be below 5s
    errors: ['rate<0.15'],              // Error rate should be below 15%
  },
};

// Counter for deterministic round-robin load balancing
let urlIndex = 0;

function getBaseUrl() {
  if (BASE_URLS.length === 1) return BASE_URLS[0];
  // Use deterministic round-robin for fair comparison
  const url = BASE_URLS[urlIndex % BASE_URLS.length];
  urlIndex++;
  return url;
}
const MAX_CPU_ITERATIONS = parseInt(__ENV.MAX_CPU_ITERATIONS || '500000');

export function setup() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('🤖 RL AUTOSCALER TEST - DYNAMIC LOAD');
  console.log('═══════════════════════════════════════════════════════');
  if (BASE_URLS.length === 1) {
    console.log(`Target: ${BASE_URLS[0]}`);
  } else {
    console.log(`Targets: ${BASE_URLS.join(', ')}`);
  }
  console.log('');
  console.log('📊 Replica Configuration:');
  console.log(`   MIN_REPLICAS: ${MIN_REPLICAS}`);
  console.log(`   MAX_REPLICAS: ${MAX_REPLICAS}`);
  console.log(`   Target Requests/Pod: ${REQUESTS_PER_POD_TARGET}`);
  console.log('');
  console.log('🚀 Dynamic VU Targets:');
  console.log(`   LOW:    ${VU_LOW} VUs`);
  console.log(`   MEDIUM: ${VU_MEDIUM} VUs`);
  console.log(`   HIGH:   ${VU_HIGH} VUs`);
  console.log(`   PEAK:   ${VU_PEAK} VUs`);
  console.log('');
  console.log('📈 Test Pattern: LOW → MEDIUM → HIGH → LOW');
  console.log('═══════════════════════════════════════════════════════\n');
}

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

export default function () {
  const currentStage = getCurrentStage(__ITER, __VU);

  // Mix of CPU and memory intensive requests (40% CPU, 60% Memory)
  // Deterministic pattern: CPU, CPU, MEM, MEM, MEM (repeats)
  const requestType = (urlIndex % 5) / 5; // Pattern: 0.0, 0.2, 0.4, 0.6, 0.8

  if (requestType < 0.4) {
    // 40% CPU-intensive requests
    // Range: 100k-500k iterations (aligned with MAX_CPU_ITERATIONS=500000)
    const iterations = 100000 + ((urlIndex * 71) % 400000); // 100k to 500k deterministic
    const safeIterations = Math.min(iterations, MAX_CPU_ITERATIONS);
    const cpuRes = safeGet(`${getBaseUrl()}/api/cpu?iterations=${safeIterations}`, { tags: { name: 'cpu', request_type: 'cpu' }, timeout: '20s' });
    cpuDuration.add(cpuRes.timings.duration);

    check(cpuRes, {
      'cpu api status is 200': (r) => r.status === 200,
      'cpu api completed': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'success';
        } catch (e) {
          return false;
        }
      },
    }) || errorRate.add(1);

  } else {
    // 50% Memory-intensive requests
    const sizeMb = 30 + ((urlIndex * 19) % 40); // 30MB to 70MB deterministic
    const memRes = safeGet(`${getBaseUrl()}/api/memory?size_mb=${sizeMb}`, { tags: { name: 'memory', request_type: 'memory' }, timeout: '20s' });
    memoryDuration.add(memRes.timings.duration);

    check(memRes, {
      'memory api status is 200': (r) => r.status === 200,
      'memory api completed': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'success';
        } catch (e) {
          return false;
        }
      },
    }) || errorRate.add(1);

  }

  requestsPerStage.add(1);

  // Variable sleep based on load phase
  const sleepTime = getSleepTime(currentStage);
  sleep(sleepTime);
}

function getCurrentStage(iteration, vu) {
  // Approximate which stage we're in based on VU count (dynamic thresholds)
  if (vu <= Math.ceil(VU_LOW * 0.5)) return 'LOW';
  if (vu <= VU_LOW) return 'MEDIUM';
  if (vu <= VU_MEDIUM) return 'HIGH';
  return 'UNKNOWN';
}

function getSleepTime(stage) {
  // Adjust sleep time based on load phase
  switch(stage) {
    case 'LOW':
      return 2 + ((urlIndex * 2) % 20) / 10; // 2-4 seconds deterministic
    case 'MEDIUM':
      return 1 + ((urlIndex * 1) % 10) / 10; // 1-2 seconds deterministic
    case 'HIGH':
      return 0.5 + ((urlIndex * 5) % 10) / 20; // 0.5-1 seconds deterministic
    default:
      return 1;
  }
}

export function handleSummary(data) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

  return {
    [`rl-autoscaler-summary-${timestamp}.json`]: JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options = {}) {
  const indent = options.indent || '';

  let summary = '\n' + indent + '🤖 RL Autoscaler Test Completed\n\n';

  // Test duration
  const duration = data.state?.testRunDurationMs / 1000 / 60 || 0;
  summary += indent + `Test Duration: ${duration.toFixed(1)} minutes\n\n`;

  // Requests summary
  summary += indent + '📊 Requests Summary:\n';
  summary += indent + `  Total Requests: ${data.metrics.http_reqs?.values.count || 0}\n`;
  summary += indent + `  Failed Requests: ${data.metrics.http_req_failed?.values.rate ?
    (data.metrics.http_req_failed.values.rate * 100).toFixed(2) : 0}%\n`;
  summary += indent + `  Success Rate: ${data.metrics.http_req_failed?.values.rate ?
    ((1 - data.metrics.http_req_failed.values.rate) * 100).toFixed(2) : 100}%\n`;

  // Response time metrics
  summary += indent + '\n⏱ Response Time:\n';
  summary += indent + `  Average: ${(data.metrics.http_req_duration?.values.avg || 0).toFixed(2)}ms\n`;
  summary += indent + `  Median (p50): ${(data.metrics.http_req_duration?.values.med || 0).toFixed(2)}ms\n`;
  summary += indent + `  95th percentile: ${(data.metrics.http_req_duration?.values['p(95)'] || 0).toFixed(2)}ms\n`;
  summary += indent + `  99th percentile: ${(data.metrics.http_req_duration?.values['p(99)'] || 0).toFixed(2)}ms\n`;
  summary += indent + `  Max: ${(data.metrics.http_req_duration?.values.max || 0).toFixed(2)}ms\n`;

  // CPU endpoint metrics
  if (data.metrics.cpu_request_duration) {
    summary += indent + '\n🔥 CPU Endpoint Performance:\n';
    summary += indent + `  Average: ${(data.metrics.cpu_request_duration.values.avg || 0).toFixed(2)}ms\n`;
    summary += indent + `  95th percentile: ${(data.metrics.cpu_request_duration.values['p(95)'] || 0).toFixed(2)}ms\n`;
  }

  // Memory endpoint metrics
  if (data.metrics.memory_request_duration) {
    summary += indent + '\n💾 Memory Endpoint Performance:\n';
    summary += indent + `  Average: ${(data.metrics.memory_request_duration.values.avg || 0).toFixed(2)}ms\n`;
    summary += indent + `  95th percentile: ${(data.metrics.memory_request_duration.values['p(95)'] || 0).toFixed(2)}ms\n`;
  }

  // Throughput
  const rps = (data.metrics.http_reqs?.values.count || 0) / ((data.state?.testRunDurationMs || 1) / 1000);
  summary += indent + `\n🚀 Throughput: ${rps.toFixed(2)} req/s\n`;

  summary += indent + '\n✅ RL Agent should have observed: LOW → MEDIUM → HIGH → LOW load pattern\n';
  summary += indent + '   Check InfluxDB/Prometheus for scaling decisions!\n';

  return summary;
}
