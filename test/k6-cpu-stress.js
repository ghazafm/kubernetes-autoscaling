import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');

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

// CPU Stress Test Configuration - Dynamic VU based on MAX_REPLICAS
export const options = {
  stages: [
    { duration: '1m', target: VU_MEDIUM },   // Ramp up to medium load
    { duration: '3m', target: VU_MEDIUM },   // Hold at medium for 3 minutes
    { duration: '1m', target: VU_PEAK },     // Spike to peak
    { duration: '2m', target: VU_PEAK },     // Hold spike
    { duration: '1m', target: 0 },           // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<9000'], // 95% of requests should be below 9s
    errors: ['rate<0.15'],             // Error rate should be below 15%
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
  console.log('🔥 CPU STRESS TEST - DYNAMIC LOAD');
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
  console.log(`   MEDIUM: ${VU_MEDIUM} VUs (initial ramp)`);
  console.log(`   PEAK:   ${VU_PEAK} VUs (spike phase)`);
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
  // Deterministic CPU load - cycles through range
  const iterations = 200000 + ((urlIndex * 67) % 600000); // 200k to 800k deterministic
  const safeIterations = Math.min(iterations, MAX_CPU_ITERATIONS);
  const cpuRes = safeGet(`${getBaseUrl()}/api/cpu?iterations=${safeIterations}`, {
    tags: { name: 'cpu', request_type: 'cpu' },
    timeout: '20s',
  });
  cpuDuration.add(cpuRes.timings.duration);

  const ok = check(cpuRes, {
    'cpu stress status is 200': (r) => r.status === 200,
    'cpu stress completed': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'success';
      } catch (e) {
        return false;
      }
    },
  });
  errorRate.add(ok ? 0 : 1);

  sleep(0.5); // Short sleep to maintain high load
}
