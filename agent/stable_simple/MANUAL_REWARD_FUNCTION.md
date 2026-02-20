# Manual Reward Function

Sumber formula: `environment/environment.py:15`

## Input
- `action` dalam rentang `0..99`
- `response_time` dalam persen SLO (`0..300` pada praktik saat ini)

## Langkah hitung manual
1. `response_time_penalty = response_time / 100`
2. `cost_penalty_raw = action / 99`
3. `cost_weight_multiplier = max(0, 1 - response_time_penalty)`
4. `effective_cost_penalty = cost_penalty_raw * cost_weight_multiplier`
5. `total_penalty = response_time_penalty + effective_cost_penalty`
6. `reward = 1 - total_penalty`

## Bentuk ringkas
`reward = 1 - (response_time/100) - (action/99) * max(0, 1 - response_time/100)`

## Piecewise (lebih mudah dicek manual)
- Jika `response_time <= 100`:
  `reward = 1 - (response_time/100) - (action/99) * (1 - response_time/100)`
- Jika `response_time > 100`:
  `reward = 1 - (response_time/100)`

## Catatan penting
- Parameter `WEIGHT_RESPONSE_TIME` dan `WEIGHT_COST` saat ini *belum* dipakai dalam operasi matematika formula reward di file tersebut.
- File tabel manual penuh ada di: `MANUAL_REWARD_FUNCTION_GRID.csv`
