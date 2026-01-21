import { sleep } from 'k6';
import exec from 'k6/execution';
import http from 'k6/http';

/*
  Enhanced k6 script for testing memory-optimized Flask application

  Key improvements:
  1) Tests both cached and non-cached memory allocations
  2) Periodically triggers manual cleanup to test malloc_trim
  3) Monitors memory stats endpoint
  4) Configurable cache behavior for testing
*/

// =====================
// 1) Parameters (env)
// =====================
const BASE_URLS = (__ENV.BASE_URLS || __ENV.BASE_URL || 'http://localhost:5000')
  .split(',')
  .map((s) => s.trim())
  .filter(Boolean);

const DURATION_MULTIPLIER = parseFloat(__ENV.DURATION_MULTIPLIER || '1');
const CYCLE_COUNT = parseInt(__ENV.CYCLE_COUNT || '1');

const MAX_REPLICAS = parseInt(__ENV.MAX_REPLICAS || '50');
const MIN_REPLICAS = parseInt(__ENV.MIN_REPLICAS || '1');
const REQUESTS_PER_POD = parseFloat(__ENV.REQUESTS_PER_POD || '8');

// Request mix (proportions)
const P_CPU = parseFloat(__ENV.P_CPU || '0.45');
const P_MEM = parseFloat(__ENV.P_MEM || '0.45'); // basic = 1 - (P_CPU + P_MEM)

// Payload configuration
const CPU_ITERATIONS = parseInt(__ENV.CPU_ITERATIONS || '400000');
const MEM_SIZE_MB = parseInt(__ENV.MEM_SIZE_MB || '30');
const REQ_TIMEOUT = __ENV.REQ_TIMEOUT || '10s';

// =====================
// NEW: Memory Management Testing Parameters
// =====================
// Percentage of memory requests that should use cache=false (to test malloc_trim)
const P_MEMORY_NO_CACHE = parseFloat(__ENV.P_MEMORY_NO_CACHE || '0.7'); // 70% non-cached

// Trigger cleanup every N iterations (to test manual cleanup)
const CLEANUP_INTERVAL = parseInt(__ENV.CLEANUP_INTERVAL || '100');

// Monitor memory stats every N iterations
const MEMORY_STATS_INTERVAL = parseInt(__ENV.MEMORY_STATS_INTERVAL || '50');

// Enable detailed memory testing
const ENABLE_MEMORY_TESTING = (__ENV.ENABLE_MEMORY_TESTING || 'true') === 'true';

// =====================
// 2) Helper functions
// =====================
function pickBaseUrl() {
  if (BASE_URLS.length === 1) return BASE_URLS[0];
  // Use random selection per request for immediate load balancing
  // This ensures balanced distribution even with low VU counts
  return BASE_URLS[Math.floor(Math.random() * BASE_URLS.length)];
}

function scaleDuration(minutes) {
  const totalMinutes = minutes * DURATION_MULTIPLIER;
  const hours = Math.floor(totalMinutes / 60);
  const mins = Math.floor(totalMinutes % 60);
  const secs = Math.floor((totalMinutes % 1) * 60);
  if (hours > 0) return mins > 0 ? `${hours}h${mins}m` : `${hours}h`;
  if (mins > 0) return secs > 0 ? `${mins}m${secs}s` : `${mins}m`;
  return `${secs}s`;
}

// =====================
// 3) Dynamic VU targets
// =====================
const VU_WARMUP = Math.ceil(MIN_REPLICAS * 2);
const VU_LOW = Math.ceil(MAX_REPLICAS * 0.2 * REQUESTS_PER_POD);
const VU_MEDIUM = Math.ceil(MAX_REPLICAS * 0.4 * REQUESTS_PER_POD);
const VU_HIGH = Math.ceil(MAX_REPLICAS * 0.6 * REQUESTS_PER_POD);
const VU_PEAK = Math.ceil(MAX_REPLICAS * 0.8 * REQUESTS_PER_POD);
const VU_SPIKE = Math.ceil(MAX_REPLICAS * 1.0 * REQUESTS_PER_POD);

const ceil = (v) => Math.ceil(v);

// =====================
// 4) Stages configuration
// =====================
const basePattern = [
  // Warm-up
  { duration: scaleDuration(1), target: 0 },
  { duration: scaleDuration(1), target: VU_WARMUP },
  { duration: scaleDuration(2), target: VU_WARMUP },

  // Phase 1: Morning ramp-up
  { duration: scaleDuration(1), target: ceil(VU_LOW * 0.5) },
  { duration: scaleDuration(2), target: ceil(VU_LOW * 0.5) },
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(2), target: VU_LOW },
  { duration: scaleDuration(1), target: ceil(VU_LOW * 1.5) },
  { duration: scaleDuration(3), target: ceil(VU_LOW * 1.5) },

  // Phase 2: Steady daytime
  { duration: scaleDuration(1), target: VU_MEDIUM },
  { duration: scaleDuration(4), target: VU_MEDIUM },

  // Phase 3: Lunch dip
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(2), target: VU_LOW },

  // Phase 4: Post-lunch recovery
  { duration: scaleDuration(1), target: ceil(VU_MEDIUM * 1.2) },
  { duration: scaleDuration(3), target: ceil(VU_MEDIUM * 1.2) },

  // Phase 5: Afternoon peak
  { duration: scaleDuration(1), target: VU_HIGH },
  { duration: scaleDuration(2), target: VU_PEAK },
  { duration: scaleDuration(4), target: VU_PEAK },

  // Phase 6: Flash spike
  { duration: scaleDuration(0.5), target: VU_SPIKE },
  { duration: scaleDuration(2), target: VU_SPIKE },
  { duration: scaleDuration(1), target: VU_PEAK },

  // Phase 7: Evening decline
  { duration: scaleDuration(1), target: VU_HIGH },
  { duration: scaleDuration(2), target: VU_MEDIUM },
  { duration: scaleDuration(2), target: ceil(VU_LOW * 1.5) },
  { duration: scaleDuration(2), target: VU_LOW },

  // Phase 8: Night-time low
  { duration: scaleDuration(1), target: ceil(VU_WARMUP * 2) },
  { duration: scaleDuration(3), target: ceil(VU_WARMUP * 2) },
  { duration: scaleDuration(1), target: VU_WARMUP },
  { duration: scaleDuration(2), target: VU_WARMUP },

  // Phase 9: Oscillating load
  { duration: scaleDuration(0.5), target: ceil(VU_LOW * 1.5) },
  { duration: scaleDuration(1), target: ceil(VU_LOW * 1.5) },
  { duration: scaleDuration(0.5), target: ceil(VU_WARMUP * 3) },
  { duration: scaleDuration(1), target: ceil(VU_WARMUP * 3) },
  { duration: scaleDuration(0.5), target: VU_MEDIUM },
  { duration: scaleDuration(1), target: VU_MEDIUM },
  { duration: scaleDuration(0.5), target: ceil(VU_WARMUP * 2) },

  // Shutdown
  { duration: scaleDuration(1), target: VU_WARMUP },
  { duration: scaleDuration(2), target: 0 },
];

function generateStages() {
  let stages = [];
  for (let i = 0; i < CYCLE_COUNT; i++) stages = stages.concat(basePattern);
  return stages;
}

export const options = {
  stages: generateStages(),
  thresholds: {
    // Monitor memory-related metrics
    'http_req_duration{endpoint:memory}': ['p(95)<2000'], // Memory endpoint should respond within 2s
    'http_req_duration{endpoint:cpu}': ['p(95)<3000'], // CPU endpoint can be slower
    'http_req_failed': ['rate<0.1'], // Less than 10% errors
  },
};

// =====================
// 5) Setup function (runs once per VU)
// =====================
export function setup() {
  console.log('=== k6 Memory Management Test Configuration ===');
  console.log(`Base URLs: ${BASE_URLS.join(', ')}`);
  console.log(`Memory allocation: ${MEM_SIZE_MB}MB`);
  console.log(`CPU iterations: ${CPU_ITERATIONS}`);
  console.log(`Request mix: ${P_CPU * 100}% CPU, ${P_MEM * 100}% Memory, ${(1 - P_CPU - P_MEM) * 100}% Basic`);
  console.log(`Memory no-cache rate: ${P_MEMORY_NO_CACHE * 100}%`);
  console.log(`Cleanup interval: every ${CLEANUP_INTERVAL} iterations`);
  console.log(`Memory stats check: every ${MEMORY_STATS_INTERVAL} iterations`);
  console.log(`Max VUs: Warmup=${VU_WARMUP}, Low=${VU_LOW}, Medium=${VU_MEDIUM}, High=${VU_HIGH}, Peak=${VU_PEAK}, Spike=${VU_SPIKE}`);
  console.log('=============================================');
}

// =====================
// 6) Main request function with memory management testing
// =====================
export default function () {
  const baseUrl = pickBaseUrl();
  const iterationInTest = exec.scenario.iterationInTest;

  // Periodically check memory stats
  if (ENABLE_MEMORY_TESTING && iterationInTest % MEMORY_STATS_INTERVAL === 0) {
    const statsRes = http.get(`${baseUrl}/memory/stats`, {
      timeout: REQ_TIMEOUT,
      tags: { endpoint: 'stats' },
    });

    if (statsRes.status === 200) {
      try {
        const stats = JSON.parse(statsRes.body);
        console.log(`[Iteration ${iterationInTest}] Memory stats: Cache=${stats.cache_count}/${stats.cache_limit}, GC=${stats.gc_stats?.count}`);
      } catch (e) {
        // Ignore parse errors
      }
    }
  }

  // Periodically trigger cleanup to test malloc_trim
  if (ENABLE_MEMORY_TESTING && iterationInTest % CLEANUP_INTERVAL === 0 && iterationInTest > 0) {
    const cleanRes = http.get(`${baseUrl}/clean`, {
      timeout: REQ_TIMEOUT,
      tags: { endpoint: 'clean' },
    });

    if (cleanRes.status === 200) {
      try {
        const result = JSON.parse(cleanRes.body);
        console.log(`[Iteration ${iterationInTest}] Cleanup: cleared=${result.cleared_items}, gc=${result.gc_collected}, trim=${result.malloc_trim_result}`);
      } catch (e) {
        // Ignore parse errors
      }
    }
  }

  // Main request - choose type based on mix
  const r = Math.random();
  let url;
  let endpoint;

  if (r < P_CPU) {
    // CPU-intensive request
    url = `${baseUrl}/api/cpu?iterations=${CPU_ITERATIONS}`;
    endpoint = 'cpu';
  } else if (r < P_CPU + P_MEM) {
    // Memory-intensive request
    // NEW: Vary between cached and non-cached to test malloc_trim
    const useCache = Math.random() > P_MEMORY_NO_CACHE;
    url = `${baseUrl}/api/memory?size_mb=${MEM_SIZE_MB}&cache=${useCache}`;
    endpoint = 'memory';
  } else {
    // Basic request
    url = `${baseUrl}/api`;
    endpoint = 'basic';
  }

  http.get(url, {
    timeout: REQ_TIMEOUT,
    tags: { endpoint: endpoint },
  });

  // Realistic pacing
  sleep(1);
}

// =====================
// 7) Teardown function (runs once at end)
// =====================
export function teardown(data) {
  console.log('=== k6 Test Complete ===');
  console.log('Triggering final cleanup on all endpoints...');

  // Trigger cleanup on all base URLs
  for (const baseUrl of BASE_URLS) {
    try {
      const cleanRes = http.get(`${baseUrl}/clean`, { timeout: '5s' });
      if (cleanRes.status === 200) {
        const result = JSON.parse(cleanRes.body);
        console.log(`Cleanup ${baseUrl}: cleared=${result.cleared_items}, gc=${result.gc_collected}, trim=${result.malloc_trim_result}`);
      }
    } catch (e) {
      console.log(`Failed to cleanup ${baseUrl}: ${e}`);
    }
  }

  console.log('========================');
}