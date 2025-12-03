import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const memoryDuration = new Trend('memory_request_duration');

// Dynamic load calculation based on replica capacity
const MAX_REPLICAS = parseInt(__ENV.MAX_REPLICAS || '50');
const MIN_REPLICAS = parseInt(__ENV.MIN_REPLICAS || '1');
const REQUESTS_PER_POD_TARGET = parseFloat(__ENV.REQUESTS_PER_POD || '8');

// Calculate VU targets to stress pods at different capacity levels
// Memory stress uses lower VU counts due to longer request duration and memory constraints
const VU_WARMUP = Math.ceil(MIN_REPLICAS * 2);  // Minimal load
const VU_LOW = Math.ceil(MAX_REPLICAS * 0.1 * REQUESTS_PER_POD_TARGET);  // 10% capacity (memory is heavier)
const VU_MEDIUM = Math.ceil(MAX_REPLICAS * 0.2 * REQUESTS_PER_POD_TARGET);  // 20% capacity

// Memory Stress Test Configuration - Dynamic VU based on MAX_REPLICAS
export const options = {
  stages: [
    { duration: '1m', target: VU_LOW },      // Ramp up to low load
    { duration: '3m', target: VU_LOW },      // Hold at low for 3 minutes
    { duration: '1m', target: VU_MEDIUM },   // Spike to medium
    { duration: '2m', target: VU_MEDIUM },   // Hold spike
    { duration: '1m', target: 0 },           // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'],  // 95% of requests should be below 5s
    errors: ['rate<0.10'],               // Error rate should be below 10%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';
const MAX_CPU_ITERATIONS = parseInt(__ENV.MAX_CPU_ITERATIONS || '500000');

export function setup() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('💾 MEMORY STRESS TEST - DYNAMIC LOAD');
  console.log('═══════════════════════════════════════════════════════');
  console.log(`Target: ${BASE_URL}`);
  console.log('');
  console.log('📊 Replica Configuration:');
  console.log(`   MIN_REPLICAS: ${MIN_REPLICAS}`);
  console.log(`   MAX_REPLICAS: ${MAX_REPLICAS}`);
  console.log(`   Target Requests/Pod: ${REQUESTS_PER_POD_TARGET}`);
  console.log('');
  console.log('🚀 Dynamic VU Targets (memory-intensive, lower VU):');
  console.log(`   LOW:    ${VU_LOW} VUs (initial ramp)`);
  console.log(`   MEDIUM: ${VU_MEDIUM} VUs (spike phase)`);
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
  // Safe memory allocation - account for concurrency (2-3 simultaneous requests)
  // Max safe: 140 MB ÷ 2 = 70 MB per request
  const sizeMb = Math.floor(Math.random() * 50) + 20; // 20MB to 70MB (safe for concurrency)
  const memRes = safeGet(`${BASE_URL}/api/memory?size_mb=${sizeMb}`, {
    tags: { name: 'memory', request_type: 'memory' },
    timeout: '20s',
  });
  memoryDuration.add(memRes.timings.duration);

  check(memRes, {
    'memory stress status is 200': (r) => r.status === 200,
    'memory stress completed': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'success';
      } catch (e) {
        return false;
      }
    },
  }) || errorRate.add(1);

  sleep(2); // Longer sleep to allow memory cleanup between requests
}
