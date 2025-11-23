import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const cpuDuration = new Trend('cpu_request_duration');
const memoryDuration = new Trend('memory_request_duration');

// Test configuration
export const options = {
  stages: [
    { duration: '30s', target: 10 },  // Ramp up to 10 users
    { duration: '1m', target: 10 },   // Stay at 10 users
    { duration: '30s', target: 20 },  // Ramp up to 20 users
    { duration: '1m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests should be below 5s
    errors: ['rate<0.1'],               // Error rate should be below 10%
  },
};

// Base URL - Update this to match your service endpoint
const BASE_URL = __ENV.BASE_URL || 'http://localhost:5000';

export default function () {
  // Test the basic /api endpoint
  const basicRes = http.get(`${BASE_URL}/api`);
  check(basicRes, {
    'basic api status is 200': (r) => r.status === 200,
  }) || errorRate.add(1);
  
  sleep(1);

  // Test CPU-intensive endpoint with varying iterations
  const iterations = Math.floor(Math.random() * 1500000) + 500000; // 500k–2.0M (stays below MAX_CPU_ITERATIONS)
  const cpuRes = http.get(`${BASE_URL}/api/cpu?iterations=${iterations}`);
  cpuDuration.add(cpuRes.timings.duration);
  
  const cpuOk = check(cpuRes, {
    'cpu api status is 200': (r) => r.status === 200,
    'cpu api has success status': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'success';
      } catch (e) {
        return false;
      }
    },
  });
  errorRate.add(cpuOk ? 0 : 1);
  
  sleep(2);

  // Test memory-intensive endpoint with varying memory sizes
  const sizeMb = Math.floor(Math.random() * 40) + 30; // Random between 30MB and 70MB (safe range)
  const memRes = http.get(`${BASE_URL}/api/memory?size_mb=${sizeMb}`);
  memoryDuration.add(memRes.timings.duration);
  
  const memOk = check(memRes, {
    'memory api status is 200': (r) => r.status === 200,
    'memory api has success status': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.status === 'success';
      } catch (e) {
        return false;
      }
    },
  });
  errorRate.add(memOk ? 0 : 1);
  
  sleep(1);
}

export function handleSummary(data) {
  return {
    'summary.json': JSON.stringify(data, null, 2),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options = {}) {
  const indent = options.indent || '';
  const enableColors = options.enableColors || false;
  
  let summary = '\n' + indent + '✓ Test completed\n\n';
  
  // Requests summary
  summary += indent + 'Requests:\n';
  summary += indent + `  Total: ${data.metrics.http_reqs?.values.count || 0}\n`;
  summary += indent + `  Failed: ${data.metrics.http_req_failed?.values.rate ? 
    (data.metrics.http_req_failed.values.rate * 100).toFixed(2) : 0}%\n`;
  
  // Duration summary
  summary += indent + '\nResponse Time:\n';
  summary += indent + `  Average: ${data.metrics.http_req_duration?.values.avg?.toFixed(2) || 0}ms\n`;
  summary += indent + `  95th percentile: ${data.metrics.http_req_duration?.values['p(95)']?.toFixed(2) || 0}ms\n`;
  summary += indent + `  Max: ${data.metrics.http_req_duration?.values.max?.toFixed(2) || 0}ms\n`;
  
  // Custom metrics
  if (data.metrics.cpu_request_duration) {
    summary += indent + '\nCPU Endpoint:\n';
    summary += indent + `  Average: ${data.metrics.cpu_request_duration.values.avg?.toFixed(2) || 0}ms\n`;
    summary += indent + `  95th percentile: ${data.metrics.cpu_request_duration.values['p(95)']?.toFixed(2) || 0}ms\n`;
  }
  
  if (data.metrics.memory_request_duration) {
    summary += indent + '\nMemory Endpoint:\n';
    summary += indent + `  Average: ${data.metrics.memory_request_duration.values.avg?.toFixed(2) || 0}ms\n`;
    summary += indent + `  95th percentile: ${data.metrics.memory_request_duration.values['p(95)']?.toFixed(2) || 0}ms\n`;
  }
  
  return summary;
}
