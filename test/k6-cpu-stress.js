import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');

// CPU Stress Test Configuration
export const options = {
  stages: [
    { duration: '1m', target: 50 },   // Ramp up to 50 concurrent users
    { duration: '3m', target: 50 },   // Hold at 50 users for 3 minutes
    { duration: '1m', target: 100 },  // Spike to 100 users
    { duration: '2m', target: 100 },  // Hold spike
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<10000'], // 95% of requests should be below 10s
    errors: ['rate<0.15'],               // Error rate should be below 15%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';

export default function () {
  // Heavy CPU load - high iteration count
  const iterations = Math.floor(Math.random() * 3000000) + 2000000; // 2M to 5M iterations
  const cpuRes = http.get(`${BASE_URL}/api/cpu?iterations=${iterations}`);
  cpuDuration.add(cpuRes.timings.duration);
  
  check(cpuRes, {
    'cpu stress status is 200': (r) => r.status === 200,
    'cpu stress completed': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'success';
      } catch (e) {
        return false;
      }
    },
  }) || errorRate.add(1);
  
  sleep(0.5); // Short sleep to maintain high load
}
