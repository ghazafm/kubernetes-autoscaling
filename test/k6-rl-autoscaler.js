import { check, sleep } from 'k6';
import http from 'k6/http';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');
const memoryDuration = new Trend('memory_request_duration');
const requestsPerStage = new Counter('requests_per_stage');

// RL Autoscaler Test Configuration
// Scenario: Low → Medium → High → Low (to test RL agent's scaling decisions)
export const options = {
  stages: [
    // Start
    { duration: '1m', target: 1 },

    // Phase 1: LOW LOAD (baseline)
    { duration: '2m', target: 5 },    // Ramp up to 5 users
    { duration: '3m', target: 5 },    // Hold at 5 users (LOW)

    // Phase 2: MEDIUM LOAD
    { duration: '1m', target: 15 },   // Ramp up to 15 users
    { duration: '4m', target: 15 },   // Hold at 15 users (MEDIUM)

    // Phase 3: HIGH LOAD
    { duration: '1m', target: 30 },   // Ramp up to 30 users
    { duration: '5m', target: 30 },   // Hold at 30 users (HIGH)

    // Phase 4: BACK TO LOW (scale down test)
    { duration: '1m', target: 5 },    // Ramp down to 5 users
    { duration: '3m', target: 5 },    // Hold at 5 users (LOW)

    // Graceful shutdown
    { duration: '30s', target: 0 },   // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests should be below 5s
    errors: ['rate<0.15'],              // Error rate should be below 15%
  },
};

// Base URL - Update this to match your service endpoint
const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';

export default function () {
  const currentStage = getCurrentStage(__ITER, __VU);

  // Mix of CPU and memory intensive requests
  const requestType = Math.random();

  if (requestType < 0.4) {
    // 40% CPU-intensive requests
    const iterations = Math.floor(Math.random() * 2000000) + 1000000; // 1M to 3M iterations
    const cpuRes = http.get(`${BASE_URL}/api/cpu?iterations=${iterations}`);
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

  } else if (requestType < 0.7) {
    // 30% Memory-intensive requests
    const sizeMb = Math.floor(Math.random() * 40) + 30; // 30MB to 70MB
    const memRes = http.get(`${BASE_URL}/api/memory?size_mb=${sizeMb}`);
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

  } else {
    // 30% Basic requests (lightweight)
    const basicRes = http.get(`${BASE_URL}/api`);

    check(basicRes, {
      'basic api status is 200': (r) => r.status === 200,
    }) || errorRate.add(1);
  }

  requestsPerStage.add(1);

  // Variable sleep based on load phase
  const sleepTime = getSleepTime(currentStage);
  sleep(sleepTime);
}

function getCurrentStage(iteration, vu) {
  // Approximate which stage we're in based on VU count
  // This is a rough estimation
  if (vu <= 5) return 'LOW';
  if (vu <= 15) return 'MEDIUM';
  if (vu <= 30) return 'HIGH';
  return 'UNKNOWN';
}

function getSleepTime(stage) {
  // Adjust sleep time based on load phase
  switch(stage) {
    case 'LOW':
      return 2 + Math.random() * 2; // 2-4 seconds
    case 'MEDIUM':
      return 1 + Math.random() * 1; // 1-2 seconds
    case 'HIGH':
      return 0.5 + Math.random() * 0.5; // 0.5-1 seconds
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
