import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const memoryDuration = new Trend('memory_request_duration');

// Memory Stress Test Configuration
export const options = {
  stages: [
    { duration: '1m', target: 5 },    // Ramp up to 5 concurrent users
    { duration: '3m', target: 5 },    // Hold at 5 users for 3 minutes
    { duration: '1m', target: 10 },   // Spike to 10 users
    { duration: '2m', target: 10 },   // Hold spike
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'],  // 95% of requests should be below 5s
    errors: ['rate<0.10'],               // Error rate should be below 10%
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
