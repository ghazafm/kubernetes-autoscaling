import { check, sleep } from 'k6';
import http from 'k6/http';
import { Counter, Gauge, Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');
const memoryDuration = new Trend('memory_request_duration');
const basicDuration = new Trend('basic_request_duration');
const requestsPerDay = new Counter('requests_per_day');
const dayOfWeekGauge = new Gauge('day_of_week');

// RL Autoscaler Weekly Simulation Test
// Simulates realistic weekly traffic patterns (compressed into ~50 minutes)
// Each "day" is ~7 minutes to make training practical
//
// CUSTOMIZABLE DURATION:
// Set DURATION_MULTIPLIER environment variable to extend test duration
// Set CYCLE_COUNT to repeat weeks multiple times
//
// Usage:
//   k6 run --env DURATION_MULTIPLIER=24 --env CYCLE_COUNT=4 k6-autoscaler-weekly.js
//   This runs 4 weeks, each week taking 24x longer = 4 weeks total

const DURATION_MULTIPLIER = parseFloat(__ENV.DURATION_MULTIPLIER || '1');
const CYCLE_COUNT = parseInt(__ENV.CYCLE_COUNT || '1');

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

// Base pattern (50 minutes = 1 week cycle)
// Now uses dynamic VU targets based on MAX_REPLICAS
const basePattern = [
  // ===== MONDAY - Week Start (Gradual Increase) =====
  // Early morning (low)
  { duration: scaleDuration(1), target: VU_WARMUP },
  // Morning rush (9 AM - users logging in)
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(1.5), target: VU_LOW },
  // Lunch dip
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.6) },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.6) },
  // Afternoon work
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.9) },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.9) },
  // Evening decline
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.4) },

  // ===== TUESDAY - Regular Business Day (Consistent) =====
  // Morning
  { duration: scaleDuration(0.5), target: ensureInt(VU_WARMUP * 1.5) },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 1.1) },
  { duration: scaleDuration(1.5), target: ensureInt(VU_LOW * 1.1) },
  // Lunch
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.7) },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.7) },
  // Afternoon (higher than Monday)
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 1.2) },
  { duration: scaleDuration(1.5), target: ensureInt(VU_LOW * 1.2) },
  // Evening
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.5) },

  // ===== WEDNESDAY - Peak Day (Highest Load) =====
  // Morning
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.4) },
  { duration: scaleDuration(1), target: ensureInt(VU_MEDIUM * 0.9) },
  { duration: scaleDuration(1.5), target: ensureInt(VU_MEDIUM * 0.9) },
  // Lunch (still high)
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.9) },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.9) },
  // Afternoon peak + marketing campaign spike
  { duration: scaleDuration(1), target: ensureInt(VU_MEDIUM * 1.1) },
  { duration: scaleDuration(1), target: ensureInt(VU_MEDIUM * 1.1) },
  // Late spike (end-of-quarter deadline)
  { duration: scaleDuration(0.5), target: VU_HIGH },
  { duration: scaleDuration(0.5), target: VU_HIGH },
  // Evening (still elevated)
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.75) },

  // ===== THURSDAY - Post-Peak (Declining) =====
  // Morning (lower than Wednesday)
  { duration: scaleDuration(0.5), target: ensureInt(VU_WARMUP * 1.75) },
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(1.5), target: VU_LOW },
  // Lunch
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.65) },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.65) },
  // Afternoon (moderate)
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.95) },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.95) },
  // Evening
  { duration: scaleDuration(0.5), target: ensureInt(VU_LOW * 0.45) },

  // ===== FRIDAY - Week End (Early Decline) =====
  // Morning (slow start)
  { duration: scaleDuration(0.5), target: VU_WARMUP },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.8) },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.8) },
  // Lunch (longer dip - people leaving early)
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.5) },
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.5) },
  // Afternoon (minimal - early finish)
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.6) },
  { duration: scaleDuration(0.5), target: ensureInt(VU_WARMUP * 1.75) },
  // Early evening drop
  { duration: scaleDuration(0.5), target: VU_WARMUP },

  // ===== SATURDAY - Weekend (Low Activity) =====
  // Late start
  { duration: scaleDuration(1), target: ensureInt(VU_WARMUP * 0.75) },
  { duration: scaleDuration(1.5), target: ensureInt(VU_WARMUP * 0.75) },
  // Midday (some activity)
  { duration: scaleDuration(1), target: ensureInt(VU_LOW * 0.4) },
  { duration: scaleDuration(1.5), target: ensureInt(VU_LOW * 0.4) },
  // Evening (minimal)
  { duration: scaleDuration(1), target: VU_WARMUP },
  { duration: scaleDuration(1), target: VU_WARMUP },

  // ===== SUNDAY - Weekend (Minimal) =====
  // Very low all day
  { duration: scaleDuration(1), target: 1 },
  { duration: scaleDuration(2), target: 1 },
  // Slight increase (planning for Monday)
  { duration: scaleDuration(1), target: VU_WARMUP },
  { duration: scaleDuration(1.5), target: VU_WARMUP },
  // Late evening prep
  { duration: scaleDuration(0.5), target: ensureInt(VU_WARMUP * 0.75) },
  { duration: scaleDuration(1), target: ensureInt(VU_WARMUP * 0.75) },

  // ===== Graceful Shutdown =====
  { duration: scaleDuration(0.5), target: 0 },
];

// Generate stages by repeating weeks
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
    http_req_duration: ['p(95)<7000'],
    'http_req_duration{expected_response:true}': ['p(99)<12000'],
    errors: ['rate<0.10'],
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
const MAX_MEMORY_MB = parseInt(__ENV.MAX_MEMORY_MB || '140');

export function setup() {
  console.log('═══════════════════════════════════════════════════════');
  console.log('📅 RL AUTOSCALER WEEKLY SIMULATION - DYNAMIC LOAD');
  console.log('═══════════════════════════════════════════════════════');
  if (BASE_URLS.length === 1) {
    console.log(`Target: ${BASE_URLS[0]}`);
  } else {
    console.log(`Targets: ${BASE_URLS.join(', ')}`);
  }
  console.log(`Duration Multiplier: ${DURATION_MULTIPLIER}x`);
  console.log(`Week Count: ${CYCLE_COUNT}`);
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

// Simulate time of day and day of week
let iterationCounter = 0;

function getDayOfWeek() {
  const totalIterations = iterationCounter++;

  // Approximate iterations per day (rough estimation)
  // Each "day" is about 7 minutes worth of iterations
  const iterationsPerDay = 200; // Adjust based on actual test

  const day = Math.floor(totalIterations / iterationsPerDay) % 7;

  const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  return days[day];
}

function getTimeOfDay(vu) {
  // Estimate time of day based on VU count (dynamic thresholds)
  if (vu <= VU_WARMUP * 0.75) return 'night';
  if (vu <= VU_WARMUP * 2) return 'early_morning';
  if (vu <= VU_LOW * 0.75) return 'morning';
  if (vu <= VU_LOW * 1.25) return 'midday';
  if (vu <= VU_MEDIUM) return 'afternoon';
  if (vu <= VU_HIGH) return 'evening_peak';
  return 'late';
}

function getTrafficPattern(dayOfWeek, timeOfDay, vu) {
  // Weekend pattern
  if (dayOfWeek === 'Saturday' || dayOfWeek === 'Sunday') {
    return {
      cpu: 0.15,
      memory: 0.10,
      basic: 0.75,
      intensity: 'light',
    };
  }

  // Weekday patterns
  if (dayOfWeek === 'Wednesday') {
    // Peak day - more intensive operations
    return {
      cpu: 0.50,
      memory: 0.30,
      basic: 0.20,
      intensity: timeOfDay === 'afternoon' || timeOfDay === 'evening_peak' ? 'maximum' : 'high',
    };
  }

  if (dayOfWeek === 'Monday') {
    // Week start - gradual ramp
    return {
      cpu: 0.35,
      memory: 0.25,
      basic: 0.40,
      intensity: timeOfDay === 'afternoon' ? 'high' : 'moderate',
    };
  }

  if (dayOfWeek === 'Friday') {
    // Week end - lighter workload
    return {
      cpu: 0.25,
      memory: 0.20,
      basic: 0.55,
      intensity: timeOfDay === 'midday' ? 'moderate' : 'light',
    };
  }

  // Tuesday, Thursday - regular business
  return {
    cpu: 0.40,
    memory: 0.25,
    basic: 0.35,
    intensity: timeOfDay === 'afternoon' ? 'high' : 'moderate',
  };
}

function getRequestParams(type, intensity) {
  if (type === 'cpu') {
    // All ranges capped to MAX_CPU_ITERATIONS (500000)
    const intensityMap = {
      light: { base: 100000, variance: 100000 },      // 100k-200k
      moderate: { base: 200000, variance: 150000 },   // 200k-350k
      high: { base: 300000, variance: 150000 },       // 300k-450k
      maximum: { base: 400000, variance: 100000 },    // 400k-500k
    };
    const params = intensityMap[intensity] || intensityMap.moderate;
    return Math.floor(params.base + Math.random() * params.variance);
  } else if (type === 'memory') {
    // All ranges capped to 70 MB for concurrency safety (MAX_MEMORY_MB=140, ÷2 for concurrent requests)
    const intensityMap = {
      light: { base: 15, variance: 15 },      // 15-30 MB
      moderate: { base: 30, variance: 20 },   // 30-50 MB
      high: { base: 45, variance: 20 },       // 45-65 MB
      maximum: { base: 50, variance: 20 },    // 50-70 MB
    };
    const params = intensityMap[intensity] || intensityMap.moderate;
    return Math.floor(params.base + Math.random() * params.variance);
  }
  return null;
}

function getSleepTime(dayOfWeek, timeOfDay, intensity) {
  let baseSleep;

  // Weekend - longer sleep
  if (dayOfWeek === 'Saturday' || dayOfWeek === 'Sunday') {
    baseSleep = 3.0 + Math.random() * 2.0; // 3-5 seconds
    return baseSleep;
  }

  // Weekday - based on time and intensity
  const timeMap = {
    night: 4.0,
    early_morning: 2.5,
    morning: 1.5,
    midday: 1.0,
    afternoon: 0.8,
    evening_peak: 0.5,
    late: 2.0,
  };

  baseSleep = timeMap[timeOfDay] || 1.0;

  // Adjust for intensity
  const intensityMultiplier = {
    light: 1.2,
    moderate: 1.0,
    high: 0.8,
    maximum: 0.6,
  };

  baseSleep *= intensityMultiplier[intensity] || 1.0;

  // Add realistic jitter
  baseSleep += Math.random() * 0.5 - 0.25;

  return Math.max(0.2, baseSleep); // Minimum 200ms
}

export default function () {
  const dayOfWeek = getDayOfWeek();
  const timeOfDay = getTimeOfDay(__VU);
  const pattern = getTrafficPattern(dayOfWeek, timeOfDay, __VU);

  // Update gauge
  dayOfWeekGauge.add(__VU);

  // Determine request type
  const rand = Math.random();
  let requestType;

  if (rand < pattern.cpu) {
    requestType = 'cpu';
  } else if (rand < pattern.cpu + pattern.memory) {
    requestType = 'memory';
  } else {
    requestType = 'basic';
  }

  // Execute request
  let res;
  const tags = {
    day_of_week: dayOfWeek,
    time_of_day: timeOfDay,
    request_type: requestType,
    intensity: pattern.intensity,
  };

  if (requestType === 'cpu') {
    let iterations = getRequestParams('cpu', pattern.intensity);
    iterations = Math.min(iterations, MAX_CPU_ITERATIONS);
    res = safeGet(`${getBaseUrl()}/api/cpu?iterations=${iterations}`, {
      tags: tags,
      timeout: '35s',
    });

    cpuDuration.add(res.timings.duration);

    check(res, {
      'cpu weekly status is 200': (r) => r.status === 200,
      'cpu weekly completed': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'success';
        } catch (e) {
          return false;
        }
      },
    }) || errorRate.add(1);

  } else if (requestType === 'memory') {
    let sizeMb = getRequestParams('memory', pattern.intensity);
    // Cap memory to configured safe max and account for concurrency
    sizeMb = Math.min(sizeMb, Math.max(1, Math.floor(MAX_MEMORY_MB / 2)));
    res = safeGet(`${getBaseUrl()}/api/memory?size_mb=${sizeMb}`, {
      tags: tags,
      timeout: '25s',
    });

    memoryDuration.add(res.timings.duration);

    check(res, {
      'memory weekly status is 200': (r) => r.status === 200,
      'memory weekly completed': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.status === 'success';
        } catch (e) {
          return false;
        }
      },
    }) || errorRate.add(1);

  } else {
    res = safeGet(`${getBaseUrl()}/api`, {
      tags: tags,
      timeout: '8s',
    });

    basicDuration.add(res.timings.duration);

    check(res, {
      'basic weekly status is 200': (r) => r.status === 200,
    }) || errorRate.add(1);
  }

  // Track requests per day
  requestsPerDay.add(1, { day: dayOfWeek });

  // Realistic sleep
  const sleepTime = getSleepTime(dayOfWeek, timeOfDay, pattern.intensity);
  sleep(sleepTime);
}

export function handleSummary(data) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').split('T')[0];
  const timeHour = new Date().toISOString().replace(/[:.]/g, '-').split('T')[1].substring(0, 5);

  return {
    [`weekly-simulation-summary-${timestamp}-${timeHour}.json`]: JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options = {}) {
  const indent = options.indent || '';

  let summary = '\n' + indent + '📅 RL AUTOSCALER WEEKLY SIMULATION TEST - RESULTS\n';
  summary += indent + '═══════════════════════════════════════════════════════\n\n';

  // Test Configuration
  summary += indent + '⚙️  TEST CONFIGURATION\n';
  summary += indent + `   Duration Multiplier: ${DURATION_MULTIPLIER}x\n`;
  summary += indent + `   Week Count: ${CYCLE_COUNT}\n`;
  summary += indent + `   Base Pattern: 50 minutes → Actual: ${(50 * DURATION_MULTIPLIER).toFixed(1)} min per week\n`;
  summary += indent + `   Total Planned Duration: ${(50 * DURATION_MULTIPLIER * CYCLE_COUNT / 60).toFixed(1)} hours (${(50 * DURATION_MULTIPLIER * CYCLE_COUNT / 60 / 24).toFixed(1)} days)\n\n`;

  const duration = (data.state?.testRunDurationMs / 1000 / 60) || 0;
  const totalRequests = data.metrics.http_reqs?.values.count || 0;
  const failedRate = data.metrics.http_req_failed?.values.rate || 0;
  const successRate = (1 - failedRate) * 100;

  summary += indent + '⏱️  TEST OVERVIEW\n';
  summary += indent + `   Duration: ${duration.toFixed(1)} minutes (${(duration/60).toFixed(1)} hours / ${(duration/60/24).toFixed(2)} days)\n`;
  summary += indent + `   Approx per "day": ${(duration/7/CYCLE_COUNT).toFixed(1)} minutes\n`;
  summary += indent + `   Total Requests: ${totalRequests.toLocaleString()}\n`;
  summary += indent + `   Avg Req/sec: ${(totalRequests / (duration * 60)).toFixed(2)}\n`;
  summary += indent + `   Success Rate: ${successRate.toFixed(2)}%\n\n`;

  summary += indent + '📊 RESPONSE TIME METRICS\n';
  summary += indent + `   Average: ${(data.metrics.http_req_duration?.values.avg || 0).toFixed(0)}ms\n`;
  summary += indent + `   Median: ${(data.metrics.http_req_duration?.values.med || 0).toFixed(0)}ms\n`;
  summary += indent + `   p90: ${(data.metrics.http_req_duration?.values['p(90)'] || 0).toFixed(0)}ms\n`;
  summary += indent + `   p95: ${(data.metrics.http_req_duration?.values['p(95)'] || 0).toFixed(0)}ms\n`;
  summary += indent + `   p99: ${(data.metrics.http_req_duration?.values['p(99)'] || 0).toFixed(0)}ms\n\n`;

  summary += indent + '📈 ENDPOINT BREAKDOWN\n';

  if (data.metrics.cpu_request_duration) {
    summary += indent + `   CPU Endpoint:\n`;
    summary += indent + `     Avg: ${(data.metrics.cpu_request_duration.values.avg || 0).toFixed(0)}ms\n`;
    summary += indent + `     p95: ${(data.metrics.cpu_request_duration.values['p(95)'] || 0).toFixed(0)}ms\n`;
  }

  if (data.metrics.memory_request_duration) {
    summary += indent + `   Memory Endpoint:\n`;
    summary += indent + `     Avg: ${(data.metrics.memory_request_duration.values.avg || 0).toFixed(0)}ms\n`;
    summary += indent + `     p95: ${(data.metrics.memory_request_duration.values['p(95)'] || 0).toFixed(0)}ms\n`;
  }

  if (data.metrics.basic_request_duration) {
    summary += indent + `   Basic Endpoint:\n`;
    summary += indent + `     Avg: ${(data.metrics.basic_request_duration.values.avg || 0).toFixed(0)}ms\n`;
    summary += indent + `     p95: ${(data.metrics.basic_request_duration.values['p(95)'] || 0).toFixed(0)}ms\n`;
  }
  summary += '\n';

  summary += indent + '📆 WEEKLY PATTERN COVERAGE\n';
  summary += indent + '   Monday: Week start - gradual increase ✓\n';
  summary += indent + '   Tuesday: Regular business day ✓\n';
  summary += indent + '   Wednesday: Peak load day + spike events ✓\n';
  summary += indent + '   Thursday: Post-peak decline ✓\n';
  summary += indent + '   Friday: Week end - early decline ✓\n';
  summary += indent + '   Saturday: Weekend - low activity ✓\n';
  summary += indent + '   Sunday: Weekend - minimal activity ✓\n\n';

  summary += indent + '🎯 DAILY TIME PATTERNS TESTED\n';
  summary += indent + '   ✓ Night-time (minimal load)\n';
  summary += indent + '   ✓ Early morning (gradual increase)\n';
  summary += indent + '   ✓ Morning rush (login spike)\n';
  summary += indent + '   ✓ Midday steady state\n';
  summary += indent + '   ✓ Lunch dip\n';
  summary += indent + '   ✓ Afternoon peak\n';
  summary += indent + '   ✓ Evening decline\n';
  summary += indent + '   ✓ Late evening wind-down\n\n';

  summary += indent + '🧠 RL AGENT WEEKLY LEARNINGS\n';
  summary += indent + '   • Different load patterns per day of week\n';
  summary += indent + '   • Peak Wednesday - mid-week optimization\n';
  summary += indent + '   • Weekend vs weekday resource allocation\n';
  summary += indent + '   • Time-of-day prediction and preemptive scaling\n';
  summary += indent + '   • Cost optimization during off-peak hours\n';
  summary += indent + '   • Handling weekly recurring patterns\n\n';

  summary += indent + '═══════════════════════════════════════════════════════\n';
  summary += indent + '📚 Weekly pattern training data collected for RL agent\n';
  summary += indent + '═══════════════════════════════════════════════════════\n\n';

  return summary;
}
