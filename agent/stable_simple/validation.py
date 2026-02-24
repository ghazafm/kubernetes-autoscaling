import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from stable_baselines3 import DQN

OBS_LOW = np.array([0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -3.0], dtype=np.float32)
OBS_HIGH = np.array([1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0], dtype=np.float32)

DEFAULT_PHASES = [
    ("warmup", 0.15, 0.25, 0.45),
    ("ramp", 0.20, 0.45, 0.75),
    ("spike", 0.15, 0.75, 1.05),
    ("sustain", 0.20, 1.05, 0.85),
    ("cooldown", 0.15, 0.85, 0.55),
    ("steady", 0.15, 0.55, 0.65),
]


def get_env_int(key: str, default: int) -> int:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_env_float(key: str, default: float) -> float:
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def find_latest_model(model_root: Path) -> Path:
    candidates = list(model_root.glob("*/final/model.zip"))
    if not candidates:
        raise FileNotFoundError(
            f"No model found under '{model_root}'. Expected paths like model/*/final/model.zip."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_sim_phases(raw: str) -> list[tuple[str, float, float, float]]:
    if not raw.strip():
        return list(DEFAULT_PHASES)

    phases: list[tuple[str, float, float, float]] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        fields = [f.strip() for f in item.split(":")]
        if len(fields) != 4:
            raise ValueError(
                f"Invalid SIM_PHASES item '{item}'. Expected format name:fraction:start:end"
            )
        name = fields[0]
        fraction = float(fields[1])
        start = float(fields[2])
        end = float(fields[3])
        if fraction <= 0:
            raise ValueError(f"Phase fraction must be > 0, got {fraction} for '{name}'")
        phases.append((name, fraction, start, end))

    if not phases:
        raise ValueError("SIM_PHASES produced an empty phase list.")
    return phases


def action_to_replicas(action: int, min_replicas: int, max_replicas: int) -> int:
    range_replicas = max(1, max_replicas - min_replicas)
    replica = int(action * range_replicas // 99 + min_replicas)
    return min(replica, max_replicas)


def calculate_reward(action: int, response_time: float) -> tuple[float, dict]:
    rt_penalty = response_time / 100.0
    cost = action / 99.0
    reward = (1.0 - rt_penalty) * (1.0 - cost)
    details = {
        "action": action,
        "response_time": response_time,
        "rt_penalty": rt_penalty,
        "cost": cost,
        "reward": reward,
    }
    return reward, details


def build_observation(
    last_observation: np.ndarray,
    action: int,
    cpu: float,
    memory: float,
    response_time: float,
) -> np.ndarray:
    action_norm = np.clip(action / 99.0, 0.0, 1.0)
    cpu_norm = np.clip(cpu / 100.0, 0.0, 1.0)
    mem_norm = np.clip(memory / 100.0, 0.0, 1.0)
    rt_norm = np.clip(response_time / 100.0, 0.0, 3.0)

    obs = np.array(
        [
            action_norm,
            cpu_norm,
            mem_norm,
            rt_norm,
            cpu_norm - float(last_observation[1]),
            mem_norm - float(last_observation[2]),
            rt_norm - float(last_observation[3]),
        ],
        dtype=np.float32,
    )
    return np.clip(obs, OBS_LOW, OBS_HIGH)


def build_workload_pattern(
    steps: int,
    phases: list[tuple[str, float, float, float]],
    wave_amplitude: float,
) -> tuple[np.ndarray, list[str]]:
    if steps <= 1:
        return np.array([0.4], dtype=np.float32), ["steady"]

    fractions = np.array([max(0.0, p[1]) for p in phases], dtype=np.float64)
    if np.sum(fractions) <= 0:
        fractions = np.ones_like(fractions)
    fractions /= np.sum(fractions)

    raw_counts = fractions * steps
    counts = np.floor(raw_counts).astype(int)
    remainder = int(steps - int(np.sum(counts)))
    if remainder > 0:
        order = np.argsort(raw_counts - counts)[::-1]
        for i in range(remainder):
            counts[order[i % len(counts)]] += 1

    pattern = np.zeros(steps, dtype=np.float32)
    phase_names: list[str] = []
    idx = 0
    for phase_idx, (name, _, start, end) in enumerate(phases):
        count = int(counts[phase_idx])
        if count <= 0:
            continue
        values = np.linspace(start, end, num=count, endpoint=False, dtype=np.float32)
        end_idx = min(idx + count, steps)
        actual_count = end_idx - idx
        pattern[idx:end_idx] = values[:actual_count]
        phase_names.extend([name] * actual_count)
        idx = end_idx
        if idx >= steps:
            break

    if idx < steps:
        last_name = phases[-1][0]
        last_value = phases[-1][3]
        pattern[idx:] = last_value
        phase_names.extend([last_name] * (steps - idx))

    for i in range(steps):
        wave = wave_amplitude * np.sin(i / 3.0)
        pattern[i] = float(np.clip(pattern[i] + wave, 0.05, 1.20))

    return pattern, phase_names[:steps]


def simulate_metrics(
    load: float,
    replicas: int,
    max_replicas: int,
    prev_cpu: float,
    prev_memory: float,
    prev_rt: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    required_replicas = 1.0 + load * max(1.0, float(max_replicas - 1))
    pressure = required_replicas / max(1.0, float(replicas))

    target_cpu = np.clip(35.0 * pressure + 8.0 * rng.normal(0.0, 1.0), 1.0, 100.0)
    target_memory = np.clip(30.0 * pressure + 6.0 * rng.normal(0.0, 1.0), 1.0, 100.0)
    target_rt = np.clip(30.0 * (pressure**1.8) + 8.0 * rng.normal(0.0, 1.0), 1.0, 300.0)

    cpu = np.clip(0.60 * prev_cpu + 0.40 * target_cpu, 1.0, 100.0)
    memory = np.clip(0.65 * prev_memory + 0.35 * target_memory, 1.0, 100.0)
    response_time = np.clip(0.45 * prev_rt + 0.55 * target_rt, 1.0, 300.0)

    return float(cpu), float(memory), float(response_time)


def top_q_actions(
    model: DQN, obs: np.ndarray, top_k: int
) -> tuple[list[tuple[int, float]], float]:
    obs_tensor = torch.tensor(
        obs.reshape(1, -1), dtype=torch.float32, device=model.device
    )
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).detach().cpu().numpy().flatten()

    top_k = int(np.clip(top_k, 1, len(q_values)))
    top_actions = np.argsort(q_values)[-top_k:][::-1]
    entries = [(int(action), float(q_values[action])) for action in top_actions]
    gap = entries[0][1] - (entries[1][1] if len(entries) > 1 else entries[0][1])
    return entries, float(gap)


def main():
    load_dotenv(".env.test")

    model_root = Path(os.getenv("MODEL_ROOT", "model"))
    model_path_env = os.getenv("MODEL_PATH", "").strip()
    min_replicas = get_env_int("MIN_REPLICAS", 1)
    max_replicas = get_env_int("MAX_REPLICAS", 10)
    sim_steps = get_env_int("SIM_STEPS", 200)
    top_k = get_env_int("TOP_K", 3)
    sim_seed = get_env_int("SIM_SEED", 42)
    wave_amplitude = get_env_float("SIM_WAVE_AMP", 0.05)
    load_multiplier = get_env_float("SIM_LOAD_MULTIPLIER", 1.0)
    phase_spec = os.getenv("SIM_PHASES", "").strip()
    start_action = 0
    device = os.getenv("DEVICE", "auto")

    if model_path_env:
        model_path = Path(model_path_env)
        if not model_path.exists():
            print(
                f"Warning: model path not found: {model_path}. "
                f"Falling back to latest model in '{model_root}'."
            )
            model_path = find_latest_model(model_root)
    else:
        model_path = find_latest_model(model_root)

    model = DQN.load(str(model_path), device=device)
    rng = np.random.default_rng(sim_seed)
    try:
        phases = parse_sim_phases(phase_spec)
    except ValueError as exc:
        print(f"Warning: {exc}. Falling back to default simulation phases.")
        phases = list(DEFAULT_PHASES)

    workload, phase_names = build_workload_pattern(
        steps=sim_steps,
        phases=phases,
        wave_amplitude=wave_amplitude,
    )
    workload_effective = np.clip(workload * load_multiplier, 0.01, 5.0).astype(
        np.float32
    )

    replicas = action_to_replicas(start_action, min_replicas, max_replicas)
    prev_obs = np.zeros(7, dtype=np.float32)

    init_cpu = get_env_float("SIM_INIT_CPU", 1.0)
    init_memory = get_env_float("SIM_INIT_MEMORY", 8.0)
    init_rt = get_env_float("SIM_INIT_RT", 0.0)

    cpu, memory, response_time = simulate_metrics(
        load=float(workload_effective[0]),
        replicas=replicas,
        max_replicas=max_replicas,
        prev_cpu=init_cpu,
        prev_memory=init_memory,
        prev_rt=init_rt,
        rng=rng,
    )
    obs = build_observation(
        last_observation=prev_obs,
        action=start_action,
        cpu=cpu,
        memory=memory,
        response_time=response_time,
    )

    print(f"Model: {model_path}")
    print(f"Device: {model.device}")
    print(
        "Simulation config: "
        f"steps={sim_steps}, start_action={start_action}, replicas_range=[{min_replicas},{max_replicas}]"
    )
    print(f"Load tuning: SIM_LOAD_MULTIPLIER={load_multiplier:.2f}")
    print(f"Simulation phases: {', '.join([p[0] for p in phases])}")
    print("")

    actions: list[int] = []
    replicas_log: list[int] = []
    load_log: list[float] = []
    rt_log: list[float] = []
    phase_log: list[str] = []

    for step in range(sim_steps):
        action_raw, _ = model.predict(obs, deterministic=True)
        action = int(action_raw)
        action = int(np.clip(action, 0, 99))

        entries, q_gap = top_q_actions(model, obs, top_k=top_k)
        replicas = action_to_replicas(action, min_replicas, max_replicas)

        next_load = float(workload_effective[min(step + 1, sim_steps - 1)])
        next_cpu, next_memory, next_rt = simulate_metrics(
            load=next_load,
            replicas=replicas,
            max_replicas=max_replicas,
            prev_cpu=cpu,
            prev_memory=memory,
            prev_rt=response_time,
            rng=rng,
        )
        reward, _ = calculate_reward(action=action, response_time=next_rt)

        next_obs = build_observation(
            last_observation=obs,
            action=action,
            cpu=next_cpu,
            memory=next_memory,
            response_time=next_rt,
        )

        top_q_text = ", ".join([f"a{a}:{q:.3f}" for a, q in entries])
        print(
            f"[step={step:03d}] phase={phase_names[step]:<8} "
            f"load={float(workload[step]):.3f}x{load_multiplier:.2f} -> {next_load:.3f} | "
            f"action={action:02d} | replicas={replicas:02d} | "
            f"cpu={next_cpu:6.2f}% mem={next_memory:6.2f}% rt={next_rt:7.2f}% | "
            f"reward={reward:+.3f} | q_gap={q_gap:.4f}"
        )
        print(f"           top_q={top_q_text}")

        actions.append(action)
        replicas_log.append(replicas)
        load_log.append(next_load)
        rt_log.append(next_rt)
        phase_log.append(phase_names[step])

        cpu, memory, response_time = next_cpu, next_memory, next_rt
        obs = next_obs

    action_arr = np.array(actions, dtype=np.float32)
    replicas_arr = np.array(replicas_log, dtype=np.float32)
    load_arr = np.array(load_log, dtype=np.float32)
    rt_arr = np.array(rt_log, dtype=np.float32)

    if np.std(action_arr) > 1e-9 and np.std(load_arr) > 1e-9:
        load_action_corr = float(np.corrcoef(load_arr, action_arr)[0, 1])
    else:
        load_action_corr = float("nan")

    rt_violations = int(np.sum(rt_arr > 100.0))
    print("")
    print("Summary:")
    print(
        f"- action: min={int(np.min(action_arr))}, max={int(np.max(action_arr))}, "
        f"mean={float(np.mean(action_arr)):.2f}"
    )
    print(
        f"- replicas: min={int(np.min(replicas_arr))}, max={int(np.max(replicas_arr))}, "
        f"mean={float(np.mean(replicas_arr)):.2f}"
    )
    print(
        f"- response_time: min={float(np.min(rt_arr)):.2f}%, max={float(np.max(rt_arr)):.2f}%, "
        f"violations(rt>100%)={rt_violations}/{len(rt_arr)}"
    )
    print(f"- correlation(load, action)={load_action_corr:.3f}")
    for phase_name in dict.fromkeys(phase_log):
        idxs = [i for i, p in enumerate(phase_log) if p == phase_name]
        if not idxs:
            continue
        phase_action_mean = float(np.mean(action_arr[idxs]))
        phase_rt_mean = float(np.mean(rt_arr[idxs]))
        print(
            f"- phase[{phase_name}] mean_action={phase_action_mean:.2f}, "
            f"mean_rt={phase_rt_mean:.2f}%"
        )


if __name__ == "__main__":
    main()
