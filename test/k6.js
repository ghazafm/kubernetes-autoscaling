import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');
const memoryDuration = new Trend('memory_request_duration');

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
const VU_SPIKE = Math.ceil(MAX_REPLICAS * 1.0 * REQUESTS_PER_POD_TARGET * URL_COUNT);

// Cap all VU calculations at VU_SPIKE to never exceed pod capacity
const vu = (v) => Math.min(Math.max(1, Math.ceil(v)), VU_SPIKE);

// Test configuration - Dynamic VU based on MAX_REPLICAS
export const options = {
  stages: [
    { duration: '30s', target: vu(VU_LOW * 0.5) },  // Ramp up slowly
    { duration: '1m', target: vu(VU_LOW * 0.5) },   // Stay at half low
    { duration: '30s', target: VU_LOW },       // Ramp up to low
    { duration: '1m', target: VU_LOW },        // Stay at low
    { duration: '30s', target: 0 },            // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests should be below 5s
    errors: ['rate<0.1'],               // Error rate should be below 10%
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
  console.log('✅ K6 BASIC TEST - DYNAMIC LOAD');
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
  // Test CPU-intensive endpoint with varying iterations
  // Range: 100k-500k iterations (aligned with MAX_CPU_ITERATIONS=500000)
  // Deterministic CPU test - cycles through range
  const iterations = 100000 + ((urlIndex * 59) % 400000); // 100k–500k deterministic
  const safeIterations = Math.min(iterations, MAX_CPU_ITERATIONS);
  const cpuRes = safeGet(`${getBaseUrl()}/api/cpu?iterations=${safeIterations}`, { tags: { name: 'cpu', request_type: 'cpu' }, timeout: '20s' });
  cpuDuration.add(cpuRes.timings.duration);

  const cpuOk = check(cpuRes, {
    'cpu api status is 200': (r) => r.status === 200,
    'cpu api has success status': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'success';
      } catch (e) {
        return false;
      }
    },
  });
  errorRate.add(cpuOk ? 0 : 1);

  sleep(2);

  // Test memory-intensive endpoint with varying memory sizes
  // Deterministic memory test - cycles through range
  const sizeMb = 30 + ((urlIndex * 23) % 40); // 30MB to 70MB deterministic
  const memRes = safeGet(`${getBaseUrl()}/api/memory?size_mb=${sizeMb}`, { tags: { name: 'memory', request_type: 'memory' }, timeout: '20s' });
  memoryDuration.add(memRes.timings.duration);

  const memOk = check(memRes, {
    'memory api status is 200': (r) => r.status === 200,
    'memory api has success status': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'success';
      } catch (e) {
        return false;
      }
    },
  });
  errorRate.add(memOk ? 0 : 1);

  sleep(1);
}

export function handleSummary(data) {
  return {
    'summary.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options = {}) {
  const indent = options.indent || '';
  const enableColors = options.enableColors || false;

  let summary = '\n' + indent + '✓ Test completed\n\n';

  // Requests summary
  summary += indent + 'Requests:\n';
  summary += indent + `  Total: ${data.metrics.http_reqs?.values.count || 0}\n`;
  summary += indent + `  Failed: ${data.metrics.http_req_failed?.values.rate ?
    (data.metrics.http_req_failed.values.rate * 100).toFixed(2) : 0}%\n`;

  // Duration summary
  summary += indent + '\nResponse Time:\n';
  summary += indent + `  Average: ${data.metrics.http_req_duration?.values.avg?.toFixed(2) || 0}ms\n`;
  summary += indent + `  95th percentile: ${data.metrics.http_req_duration?.values['p(95)']?.toFixed(2) || 0}ms\n`;
  summary += indent + `  Max: ${data.metrics.http_req_duration?.values.max?.toFixed(2) || 0}ms\n`;

  // Custom metrics
  if (data.metrics.cpu_request_duration) {
    summary += indent + '\nCPU Endpoint:\n';
    summary += indent + `  Average: ${data.metrics.cpu_request_duration.values.avg?.toFixed(2) || 0}ms\n`;
    summary += indent + `  95th percentile: ${data.metrics.cpu_request_duration.values['p(95)']?.toFixed(2) || 0}ms\n`;
  }

  if (data.metrics.memory_request_duration) {
    summary += indent + '\nMemory Endpoint:\n';
    summary += indent + `  Average: ${data.metrics.memory_request_duration.values.avg?.toFixed(2) || 0}ms\n`;
    summary += indent + `  95th percentile: ${data.metrics.memory_request_duration.values['p(95)']?.toFixed(2) || 0}ms\n`;
  }

  return summary;
}
