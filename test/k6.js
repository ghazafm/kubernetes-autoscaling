import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');
const memoryDuration = new Trend('memory_request_duration');

// Dynamic load calculation based on replica capacity
const MAX_REPLICAS = parseInt(__ENV.MAX_REPLICAS || '50');
const MIN_REPLICAS = parseInt(__ENV.MIN_REPLICAS || '1');
const REQUESTS_PER_POD_TARGET = parseFloat(__ENV.REQUESTS_PER_POD || '8');

// Calculate VU targets to stress pods at different capacity levels
const VU_WARMUP = Math.ceil(MIN_REPLICAS * 2);  // Minimal load
const VU_LOW = Math.ceil(MAX_REPLICAS * 0.2 * REQUESTS_PER_POD_TARGET);  // 20% capacity
const VU_MEDIUM = Math.ceil(MAX_REPLICAS * 0.4 * REQUESTS_PER_POD_TARGET);  // 40% capacity

// Test configuration - Dynamic VU based on MAX_REPLICAS
export const options = {
  stages: [
    { duration: '30s', target: VU_WARMUP },   // Ramp up to warmup
    { duration: '1m', target: VU_WARMUP },    // Stay at warmup
    { duration: '30s', target: VU_LOW },      // Ramp up to low
    { duration: '1m', target: VU_LOW },       // Stay at low
    { duration: '30s', target: 0 },           // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests should be below 5s
    errors: ['rate<0.1'],               // Error rate should be below 10%
  },
};

// Base URL - Update this to match your service endpoint
const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';
const MAX_CPU_ITERATIONS = parseInt(__ENV.MAX_CPU_ITERATIONS || '500000');

export function setup() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('✅ K6 BASIC TEST - DYNAMIC LOAD');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`Target: ${BASE_URL}`);
  console.log('');
  console.log('📊 Replica Configuration:');
  console.log(`   MIN_REPLICAS: ${MIN_REPLICAS}`);
  console.log(`   MAX_REPLICAS: ${MAX_REPLICAS}`);
  console.log(`   Target Requests/Pod: ${REQUESTS_PER_POD_TARGET}`);
  console.log('');
  console.log('🚀 Dynamic VU Targets:');
  console.log(`   WARMUP: ${VU_WARMUP} VUs`);
  console.log(`   LOW:    ${VU_LOW} VUs`);
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
  // Test the basic /api endpoint
  const basicRes = safeGet(`${BASE_URL}/api`, { tags: { name: 'basic', request_type: 'basic' }, timeout: '5s' });
  check(basicRes, {
    'basic api status is 200': (r) => r.status === 200,
  }) || errorRate.add(1);

  sleep(1);

  // Test CPU-intensive endpoint with varying iterations
  // Range: 100k-500k iterations (aligned with MAX_CPU_ITERATIONS=500000)
  const iterations = Math.floor(Math.random() * 400000) + 100000; // 100k–500k iterations
  const safeIterations = Math.min(iterations, MAX_CPU_ITERATIONS);
  const cpuRes = safeGet(`${BASE_URL}/api/cpu?iterations=${safeIterations}`, { tags: { name: 'cpu', request_type: 'cpu' }, timeout: '20s' });
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
  const sizeMb = Math.floor(Math.random() * 40) + 30; // Random between 30MB and 70MB (safe range)
  const memRes = safeGet(`${BASE_URL}/api/memory?size_mb=${sizeMb}`, { tags: { name: 'memory', request_type: 'memory' }, timeout: '20s' });
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
