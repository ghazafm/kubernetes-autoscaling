import { sleep } from 'k6';
import exec from 'k6/execution';
import http from 'k6/http';

/*
  Enhanced k6 script for fair comparison testing between RL Agent and HPA

  CRITICAL: This script uses DETERMINISTIC load patterns for fair comparison

  Key features for fairness:
  1) Deterministic round-robin URL selection - ensures both deployments get identical load
  2) Deterministic request type selection - based on iteration number, not random
  3) Deterministic cache behavior - predictable pattern for memory testing
  4) Both systems receive EXACTLY the same request sequence

  Additional features:
  - Tests both cached and non-cached memory allocations
  - Periodically triggers manual cleanup to test malloc_trim
  - Monitors memory stats endpoint
  - Configurable cache behavior for testing
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
const P_CPU = parseFloat(__ENV.P_CPU || '0.5');
const P_MEM = parseFloat(__ENV.P_MEM || '0.5');

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
// Counter for deterministic round-robin load balancing
let urlIndex = 0;

function pickBaseUrl() {
  if (BASE_URLS.length === 1) return BASE_URLS[0];
  // Use deterministic round-robin to ensure IDENTICAL load distribution
  // This is CRITICAL for fair comparison between HPA and RL Agent
  const url = BASE_URLS[urlIndex % BASE_URLS.length];
  urlIndex++;
  return url;
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
// Formula: VUs = replicas * requests_per_pod * url_count * utilization_factor
const URL_COUNT = BASE_URLS.length;
const VU_LOW = Math.max(1, Math.ceil(MAX_REPLICAS * 0.2 * REQUESTS_PER_POD * URL_COUNT));
const VU_MEDIUM = Math.ceil(MAX_REPLICAS * 0.4 * REQUESTS_PER_POD * URL_COUNT);
const VU_HIGH = Math.ceil(MAX_REPLICAS * 0.6 * REQUESTS_PER_POD * URL_COUNT);
const VU_PEAK = Math.ceil(MAX_REPLICAS * 0.8 * REQUESTS_PER_POD * URL_COUNT);
const VU_SPIKE = Math.ceil(MAX_REPLICAS * 1.0 * REQUESTS_PER_POD * URL_COUNT);

// Cap all VU calculations at VU_SPIKE to never exceed pod capacity
const vu = (v) => Math.min(Math.max(1, Math.ceil(v)), VU_SPIKE);

// =====================
// 4) Stages configuration
// =====================
// Total duration: exactly 60 minutes (1 hour)
// Warm-up: 5min, Phase1: 8min, Phase2: 5min, Phase3: 3min, Phase4: 3min,
// Phase5: 6min, Phase6: 3min, Phase7: 6min, Phase8: 6min, Phase9: 5min, Shutdown: 10min = 60min
const basePattern = [
  // Warm-up with extended 0 VU (let agent scale to minimum) - 5 min
  { duration: scaleDuration(3), target: 0 },
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.5) },
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.5) },

  // Phase 1: Morning ramp-up - 8 min
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.5) },
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.5) },
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(2), target: VU_LOW },
  { duration: scaleDuration(1), target: vu(VU_LOW * 1.5) },
  { duration: scaleDuration(2), target: vu(VU_LOW * 1.5) },

  // Phase 2: Steady daytime - 5 min
  { duration: scaleDuration(1), target: VU_MEDIUM },
  { duration: scaleDuration(4), target: VU_MEDIUM },

  // Phase 3: Lunch dip - 3 min
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(2), target: VU_LOW },

  // Phase 4: Post-lunch recovery - 3 min
  { duration: scaleDuration(1), target: vu(VU_MEDIUM * 1.2) },
  { duration: scaleDuration(2), target: vu(VU_MEDIUM * 1.2) },

  // Phase 5: Afternoon peak - 6 min
  { duration: scaleDuration(1), target: VU_HIGH },
  { duration: scaleDuration(2), target: VU_PEAK },
  { duration: scaleDuration(3), target: VU_PEAK },

  // Phase 6: Flash spike - 3 min
  { duration: scaleDuration(0.5), target: VU_SPIKE },
  { duration: scaleDuration(1.5), target: VU_SPIKE },
  { duration: scaleDuration(1), target: VU_PEAK },

  // Phase 7: Evening decline - 6 min
  { duration: scaleDuration(1), target: VU_HIGH },
  { duration: scaleDuration(1), target: VU_MEDIUM },
  { duration: scaleDuration(2), target: vu(VU_LOW * 1.5) },
  { duration: scaleDuration(2), target: VU_LOW },

  // Phase 8: Night-time low - 6 min
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(2), target: VU_LOW },
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.5) },
  { duration: scaleDuration(2), target: vu(VU_LOW * 0.5) },

  // Phase 9: Oscillating load - 5 min
  { duration: scaleDuration(0.5), target: vu(VU_LOW * 1.5) },
  { duration: scaleDuration(1), target: vu(VU_LOW * 1.5) },
  { duration: scaleDuration(0.5), target: VU_LOW },
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(0.5), target: VU_MEDIUM },
  { duration: scaleDuration(1), target: VU_MEDIUM },
  { duration: scaleDuration(0.5), target: VU_LOW },

  // Shutdown with extended 0 VU (let agent scale to minimum) - 10 min
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.5) },
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.25) },
  { duration: scaleDuration(8), target: 0 },
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
  console.log('=== k6 Fair Comparison Test Configuration ===');
  console.log(`Base URLs: ${BASE_URLS.join(', ')}`);
  console.log(`Load balancing: DETERMINISTIC round-robin (fair comparison)`);
  console.log(`Request pattern: DETERMINISTIC based on iteration (no randomness)`);
  console.log(`Memory allocation: ${MEM_SIZE_MB}MB`);
  console.log(`CPU iterations: ${CPU_ITERATIONS}`);
  console.log(`Request mix: ${P_CPU * 100}% CPU, ${P_MEM * 100}% Memory`);
  console.log(`Memory no-cache rate: ${P_MEMORY_NO_CACHE * 100}% (deterministic pattern)`);
  console.log(`Cleanup interval: every ${CLEANUP_INTERVAL} iterations`);
  console.log(`Memory stats check: every ${MEMORY_STATS_INTERVAL} iterations`);
  console.log(`Max VUs: Low=${VU_LOW}, Medium=${VU_MEDIUM}, High=${VU_HIGH}, Peak=${VU_PEAK}, Spike=${VU_SPIKE}`);
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

  // Main request - choose type based on mix (50/50 CPU vs Memory)
  // Use deterministic alternating pattern: CPU, Memory, CPU, Memory...
  const r = (iterationInTest % 2) / 2; // Alternates: 0.0, 0.5, 0.0, 0.5...
  let url;
  let endpoint;

  if (r < P_CPU) {
    // CPU-intensive request
    url = `${baseUrl}/api/cpu?iterations=${CPU_ITERATIONS}`;
    endpoint = 'cpu';
  } else {
    // Memory-intensive request
    // Use deterministic cache pattern for fair comparison
    const useCache = (iterationInTest % 10) >= 7; // 70% non-cached (0-6), 30% cached (7-9)
    url = `${baseUrl}/api/memory?size_mb=${MEM_SIZE_MB}&cache=${useCache}`;
    endpoint = 'memory';
  }

  http.get(url, {
    timeout: REQ_TIMEOUT,
    tags: { endpoint: endpoint },
  });

  // Realistic pacing
  sleep(1);
}