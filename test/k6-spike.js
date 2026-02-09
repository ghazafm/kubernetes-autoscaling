import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

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
const VU_SPIKE = Math.ceil(MAX_REPLICAS * 1.0 * REQUESTS_PER_POD_TARGET * URL_COUNT);

// Spike Test - Sudden burst of traffic (Dynamic VU based on MAX_REPLICAS)
export const options = {
  stages: [
    { duration: '30s', target: VU_LOW },       // Start with low load
    { duration: '10s', target: VU_SPIKE },     // Sudden spike to max capacity
    { duration: '1m', target: VU_SPIKE },      // Hold spike
    { duration: '30s', target: VU_LOW },       // Drop back down
    { duration: '30s', target: 0 },            // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<15000'], // Allow more time during spike
    errors: ['rate<0.2'],               // Allow higher error rate during spike
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
  console.log(`   LOW:   ${VU_LOW} VUs (baseline)`);
  console.log(`   SPIKE: ${VU_SPIKE} VUs (100% capacity burst)`);
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
  // Deterministic endpoint selection for fair comparison
  const endpoint = (urlIndex % 2) === 0 ? 'cpu' : 'memory';

  let url;
  if (endpoint === 'cpu') {
    // Deterministic CPU iterations (cycles through range)
    const iterations = 200000 + ((urlIndex * 47) % 400000); // 200k-600k deterministic
    const safeIterations = Math.min(iterations, MAX_CPU_ITERATIONS);
    url = `${getBaseUrl()}/api/cpu?iterations=${safeIterations}`;
  } else {
    // Deterministic memory size (cycles through range)
    const sizeMb = 20 + ((urlIndex * 13) % 50); // 20-70 MB deterministic
    url = `${getBaseUrl()}/api/memory?size_mb=${sizeMb}`;
  }

  const res = safeGet(url, { tags: { name: (url.includes('/api/cpu') ? 'cpu' : 'memory'), request_type: (url.includes('/api/cpu') ? 'cpu' : 'memory') }, timeout: '20s' });

  const ok = check(res, {
    'spike test status is 200': (r) => r.status === 200,
  });
  errorRate.add(ok ? 0 : 1);

  sleep(0.5);
}
