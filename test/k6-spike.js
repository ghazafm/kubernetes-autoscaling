import { check, sleep } from 'k6';
import http from 'k6/http';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

// Spike Test - Sudden burst of traffic
export const options = {
  stages: [
    { duration: '30s', target: 5 },    // Start with 5 users
    { duration: '10s', target: 70 },   // Sudden spike to ~70 users (fits 5x500m)
    { duration: '1m', target: 70 },    // Hold spike
    { duration: '30s', target: 5 },    // Drop back down
    { duration: '30s', target: 0 },    // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<15000'], // Allow more time during spike
    errors: ['rate<0.2'],               // Allow higher error rate during spike
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';

export default function () {
  const endpoint = Math.random() > 0.5 ? 'cpu' : 'memory';

  let url;
  if (endpoint === 'cpu') {
    // Keep CPU load within safe range for 500m limit and MAX_CPU_ITERATIONS=2.5M
    const iterations = Math.floor(Math.random() * 400000) + 200000; // 200k-600k
    url = `${BASE_URL}/api/cpu?iterations=${iterations}`;
  } else {
    // FIXED: Max 70 MB to account for concurrent requests (was 50-200 MB!)
    const sizeMb = Math.floor(Math.random() * 50) + 20; // 20-70 MB
    url = `${BASE_URL}/api/memory?size_mb=${sizeMb}`;
  }

  const res = http.get(url);

  const ok = check(res, {
    'spike test status is 200': (r) => r.status === 200,
  });
  errorRate.add(ok ? 0 : 1);

  sleep(0.5);
}
