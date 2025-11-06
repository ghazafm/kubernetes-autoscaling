import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

// Spike Test - Sudden burst of traffic
export const options = {
  stages: [
    { duration: '30s', target: 5 },    // Start with 5 users
    { duration: '10s', target: 200 },  // Sudden spike to 200 users
    { duration: '1m', target: 200 },   // Hold spike
    { duration: '30s', target: 5 },    // Drop back down
    { duration: '30s', target: 0 },    // Ramp down to 0
  ],
  thresholds: {
    http_req_duration: ['p(95)<15000'], // Allow more time during spike
    errors: ['rate<0.2'],                // Allow higher error rate during spike
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';

export default function () {
  const endpoint = Math.random() > 0.5 ? 'cpu' : 'memory';
  
  let url;
  if (endpoint === 'cpu') {
    const iterations = Math.floor(Math.random() * 1500000) + 500000;
    url = `${BASE_URL}/api/cpu?iterations=${iterations}`;
  } else {
    const sizeMb = Math.floor(Math.random() * 150) + 50;
    url = `${BASE_URL}/api/memory?size_mb=${sizeMb}`;
  }
  
  const res = http.get(url);
  
  check(res, {
    'spike test status is 200': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  sleep(0.5);
}
