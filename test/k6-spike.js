import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

// Dynamic load calculation based on replica capacity
const MAX_REPLICAS = parseInt(__ENV.MAX_REPLICAS || '50');
const MIN_REPLICAS = parseInt(__ENV.MIN_REPLICAS || '1');
const REQUESTS_PER_POD_TARGET = parseFloat(__ENV.REQUESTS_PER_POD || '8');

// Calculate VU targets to stress pods at different capacity levels
const VU_WARMUP = Math.ceil(MIN_REPLICAS * 2);  // Minimal load
const VU_LOW = Math.ceil(MAX_REPLICAS * 0.2 * REQUESTS_PER_POD_TARGET);  // 20% capacity
const VU_SPIKE = Math.ceil(MAX_REPLICAS * 1.0 * REQUESTS_PER_POD_TARGET);  // 100% capacity

// Spike Test - Sudden burst of traffic (Dynamic VU based on MAX_REPLICAS)
export const options = {
  stages: [
    { duration: '30s', target: VU_WARMUP },    // Start with warmup
    { duration: '10s', target: VU_SPIKE },     // Sudden spike to max capacity
    { duration: '1m', target: VU_SPIKE },      // Hold spike
    { duration: '30s', target: VU_WARMUP },    // Drop back down
    { duration: '30s', target: 0 },            // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<15000'], // Allow more time during spike
    errors: ['rate<0.2'],               // Allow higher error rate during spike
  },
};

// Support multiple target URLs via BASE_URLS (comma-separated) or single BASE_URL
const BASE_URLS_RAW = __ENV.BASE_URLS || __ENV.BASE_URL || 'http://localhost:5000';
const BASE_URLS = BASE_URLS_RAW.split(',').map(s => s.trim()).filter(Boolean);

function getBaseUrl() {
  if (BASE_URLS.length === 1) return BASE_URLS[0];
  // Use random selection per request for immediate load balancing
  // This ensures balanced distribution even with low VU counts
  return BASE_URLS[Math.floor(Math.random() * BASE_URLS.length)];
}
const MAX_CPU_ITERATIONS = parseInt(__ENV.MAX_CPU_ITERATIONS || '500000');

export function setup() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('⚡ SPIKE TEST - DYNAMIC LOAD');
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
  console.log(`   WARMUP: ${VU_WARMUP} VUs (baseline)`);
  console.log(`   SPIKE:  ${VU_SPIKE} VUs (100% capacity burst)`);
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
  const endpoint = Math.random() > 0.5 ? 'cpu' : 'memory';

  let url;
  if (endpoint === 'cpu') {
    // Keep CPU load within safe range for 500m limit and MAX_CPU_ITERATIONS=2.5M
    const iterations = Math.floor(Math.random() * 400000) + 200000; // 200k-600k
    const safeIterations = Math.min(iterations, MAX_CPU_ITERATIONS);
    url = `${getBaseUrl()}/api/cpu?iterations=${safeIterations}`;
  } else {
    // FIXED: Max 70 MB to account for concurrent requests (was 50-200 MB!)
    const sizeMb = Math.floor(Math.random() * 50) + 20; // 20-70 MB
    url = `${getBaseUrl()}/api/memory?size_mb=${sizeMb}`;
  }

  const res = safeGet(url, { tags: { name: (url.includes('/api/cpu') ? 'cpu' : 'memory'), request_type: (url.includes('/api/cpu') ? 'cpu' : 'memory') }, timeout: '20s' });

  const ok = check(res, {
    'spike test status is 200': (r) => r.status === 200,
  });
  errorRate.add(ok ? 0 : 1);

  sleep(0.5);
}
