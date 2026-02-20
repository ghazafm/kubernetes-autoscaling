# Manualisasi Dua Reward Function

File ini berisi perhitungan manual untuk:
- Fungsi A: reward bawaan di `environment/environment.py:15`
- Fungsi B: reward linear alternatif

## 1) Fungsi A (Piecewise - Existing)

Referensi: `environment/environment.py:15`

Input:
- `action` dalam rentang `0..99`
- `response_time` dalam persen (`0..100+`)
- `weight_response_time` (`w_rt`)
- `weight_cost` (`w_cost`)

Konstanta:
- `RESPONSE_TIME_HIGH_THRESHOLD = 50.0`
- `RESPONSE_TIME_VIOLATION_THRESHOLD = 80.0`

Langkah hitung:
1. Hitung `rt_penalty`:
   - Jika `response_time <= 50`: `rt_penalty = 0.0`
   - Jika `50 < response_time <= 80`: `rt_penalty = (response_time - 50) / 30`
   - Jika `response_time > 80`: `rt_penalty = 1.0 + ((response_time - 80) / 80)`
2. Hitung `cost_raw = action / 99.0`
3. Hitung `cost_mult`:
   - Jika `response_time > 80`: `cost_mult = 0.0`
   - Jika `response_time <= 50`: `cost_mult = 1.0`
   - Jika `50 < response_time <= 80`: `cost_mult = 1.0 - rt_penalty`
4. Hitung `cost_eff = cost_raw * cost_mult`
5. Hitung `total_penalty = (w_rt * rt_penalty) + (w_cost * cost_eff)`
6. Hitung `reward = 1.0 - total_penalty`

Template manual Fungsi A:

| action | response_time | w_rt | w_cost | rt_penalty | cost_raw | cost_mult | cost_eff | total_penalty | reward |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |

## 2) Fungsi B (Linear - Alternatif)

Rumus usulan:

```python
response_time_penalty = response_time / 100.0
cost_penalty_raw = action / 99.0
cost_weight_multiplier = max(0.0, 1.0 - response_time_penalty)
effective_cost_penalty = cost_penalty_raw * cost_weight_multiplier
total_penalty = response_time_penalty + effective_cost_penalty
reward = 1.0 - total_penalty
```

Langkah hitung:
1. Hitung `response_time_penalty = response_time / 100.0`
2. Hitung `cost_penalty_raw = action / 99.0`
3. Hitung `cost_weight_multiplier = max(0.0, 1.0 - response_time_penalty)`
4. Hitung `effective_cost_penalty = cost_penalty_raw * cost_weight_multiplier`
5. Hitung `total_penalty = response_time_penalty + effective_cost_penalty`
6. Hitung `reward = 1.0 - total_penalty`

Template manual Fungsi B:

| action | response_time | response_time_penalty | cost_penalty_raw | cost_weight_multiplier | effective_cost_penalty | total_penalty | reward |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |

## 3) Contoh Hitung Cepat (w_rt=1.0, w_cost=1.0 untuk Fungsi A)

### Contoh 1
Input: `action=20`, `response_time=40`
- Fungsi A: `reward = 0.7980`
- Fungsi B: `reward = 0.4788`

### Contoh 2
Input: `action=60`, `response_time=65`
- Fungsi A: `reward = 0.1970`
- Fungsi B: `reward = 0.1379`

### Contoh 3
Input: `action=80`, `response_time=100`
- Fungsi A: `reward = -0.2500`
- Fungsi B: `reward = 0.0000`

## 4) Kondisi Reward Ekstrem (Maksimum, Minimum, Nol)

Asumsi untuk Fungsi A: `w_rt=1.0`, `w_cost=1.0`.

### Fungsi A (existing)
- Reward maksimum: `1.0000` pada `response_time <= 50` dan `action=0`
- Reward minimum:
  - Teoritis: `-> -inf` saat `response_time -> inf`
  - Pada rentang uji `response_time <= 300`: minimum `-2.7500` pada `response_time=300`
- Reward `0`:
  - `action=99` dan `response_time <= 80`, atau
  - `response_time=80` untuk action apa pun

### Fungsi B (alternatif linear)
- Reward maksimum: `1.0000` pada `response_time=0` dan `action=0`
- Reward minimum:
  - Teoritis: `-> -inf` saat `response_time -> inf`
  - Pada rentang uji `response_time <= 300`: minimum `-2.0000` pada `response_time=300`
- Reward `0`:
  - Untuk `0 <= response_time < 100`: saat `action=99`
  - Untuk `response_time=100`: action apa pun
