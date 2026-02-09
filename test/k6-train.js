import { sleep } from 'k6';
import http from 'k6/http';

/*
  Skrip k6 ini merealisasikan:
  1) Pola beban harian melalui stages (target VU vs waktu)
  2) Heterogenitas permintaan melalui request mix: CPU-intensif / memori-intensif

  Parameter utama dikontrol via environment variables (__ENV) agar eksperimen replikatif.
*/

// =====================
// 1) Parameter (env)
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

// request mix (proporsi) -> selaras dengan Gambar request mix di skripsi
const P_CPU = parseFloat(__ENV.P_CPU || '0.5');
const P_MEM = parseFloat(__ENV.P_MEM || '0.5'); // CPU and memory only

// payload sederhana (konstan) agar kode ringkas
const CPU_ITERATIONS = parseInt(__ENV.CPU_ITERATIONS || '400000');
const MEM_SIZE_MB = parseInt(__ENV.MEM_SIZE_MB || '30');
const REQ_TIMEOUT = __ENV.REQ_TIMEOUT || '10s';

// =====================
// 2) Helper
// =====================
// Counter for deterministic round-robin load balancing
let urlIndex = 0;

function getBaseUrl() {
  if (BASE_URLS.length === 1) return BASE_URLS[0];
  // Use deterministic round-robin for fair comparison
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
// 3) Dynamic VU target (tetap sama logika)
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
// 4) Stages: duration & target SAMA seperti skrip kamu
// =====================
const basePattern = [
  // Warm-up
  { duration: scaleDuration(3), target: 0 },

  // Phase 1: Morning ramp-up
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.5) },
  { duration: scaleDuration(2), target: vu(VU_LOW * 0.5) },
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(2), target: VU_LOW },
  { duration: scaleDuration(1), target: vu(VU_LOW * 1.5) },
  { duration: scaleDuration(3), target: vu(VU_LOW * 1.5) },

  // Phase 2: Steady daytime
  { duration: scaleDuration(1), target: VU_MEDIUM },
  { duration: scaleDuration(4), target: VU_MEDIUM },

  // Phase 3: Lunch dip
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(2), target: VU_LOW },

  // Phase 4: Post-lunch recovery
  { duration: scaleDuration(1), target: vu(VU_MEDIUM * 1.2) },
  { duration: scaleDuration(3), target: vu(VU_MEDIUM * 1.2) },

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
  { duration: scaleDuration(2), target: vu(VU_LOW * 1.5) },
  { duration: scaleDuration(2), target: VU_LOW },

  // Phase 8: Night-time low
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(3), target: VU_LOW },
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.5) },
  { duration: scaleDuration(2), target: vu(VU_LOW * 0.5) },

  // Phase 9: Oscillating load
  { duration: scaleDuration(0.5), target: vu(VU_LOW * 1.5) },
  { duration: scaleDuration(1), target: vu(VU_LOW * 1.5) },
  { duration: scaleDuration(0.5), target: VU_LOW },
  { duration: scaleDuration(1), target: VU_LOW },
  { duration: scaleDuration(0.5), target: VU_MEDIUM },
  { duration: scaleDuration(1), target: VU_MEDIUM },
  { duration: scaleDuration(0.5), target: VU_LOW },

  // Shutdown
  { duration: scaleDuration(1), target: vu(VU_LOW * 0.5) },
  { duration: scaleDuration(2), target: 0 },
];

function generateStages() {
  let stages = [];
  for (let i = 0; i < CYCLE_COUNT; i++) stages = stages.concat(basePattern);
  return stages;
}

// `stages` adalah daftar {duration, target} untuk meramp jumlah VU. :contentReference[oaicite:1]{index=1}
export const options = {
  stages: generateStages(),
};

// =====================
// 5) Default function: request mix (cpu/memory)
// =====================
export default function () {
  const baseUrl = pickBaseUrl();

  // pilih tipe request sesuai proporsi (request mix)
  // Deterministic alternating pattern: CPU, Memory, CPU, Memory...
  const r = (urlIndex % 2) / 2; // Alternates: 0.0, 0.5, 0.0, 0.5...
  let url;

  if (r < P_CPU) {
    url = `${baseUrl}/api/cpu?iterations=${CPU_ITERATIONS}`;
  } else {
    url = `${baseUrl}/api/memory?size_mb=${MEM_SIZE_MB}`;
  }

  http.get(url, { timeout: REQ_TIMEOUT });

  // pacing sederhana agar tiap VU tidak menembak request tanpa jeda (lebih realistis)
  sleep(1);
}
