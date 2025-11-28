import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');

// CPU Stress Test Configuration
export const options = {
  stages: [
    { duration: '1m', target: 40 },   // Ramp up to 40 concurrent users
    { duration: '3m', target: 40 },   // Hold at 40 users for 3 minutes
    { duration: '1m', target: 80 },   // Spike to 80 users (aligned with 5x500m pods)
    { duration: '2m', target: 80 },   // Hold spike
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<9000'], // 95% of requests should be below 9s
    errors: ['rate<0.15'],             // Error rate should be below 15%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';
const MAX_CPU_ITERATIONS = parseInt(__ENV.MAX_CPU_ITERATIONS || '2500000');

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
  // Heavy CPU load - high iteration count
  const iterations = Math.floor(Math.random() * 600000) + 200000; // 200k to 800k iterations
  const safeIterations = Math.min(iterations, MAX_CPU_ITERATIONS);
  const cpuRes = safeGet(`${BASE_URL}/api/cpu?iterations=${safeIterations}`, {
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
