import argparse
import csv
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

OUT_COLUMNS = [
    "timestamp",
    "episode",
    "step",
    "obs_action",
    "obs_cpu",
    "obs_memory",
    "obs_response_time",
    "obs_cpu_delta",
    "obs_memory_delta",
    "obs_rt_delta",
    "action",
    "reward",
    "next_obs_action",
    "next_obs_cpu",
    "next_obs_memory",
    "next_obs_response_time",
    "next_obs_cpu_delta",
    "next_obs_memory_delta",
    "next_obs_rt_delta",
    "terminated",
    "truncated",
    "cpu",
    "memory",
    "response_time",
    "replicas",
]

CURRENT_4D_REQUIRED = {
    "timestamp",
    "episode",
    "step",
    "obs_action",
    "obs_cpu",
    "obs_memory",
    "obs_response_time",
    "action",
    "reward",
    "next_obs_action",
    "next_obs_cpu",
    "next_obs_memory",
    "next_obs_response_time",
    "terminated",
    "truncated",
    "cpu",
    "memory",
    "response_time",
    "replicas",
}

LEGACY_4D_REQUIRED = {
    "timestamp",
    "episode",
    "step",
    "obs_action",
    "obs_cpu_relative",
    "obs_memory_relative",
    "obs_response_time",
    "action",
    "reward",
    "next_obs_action",
    "next_obs_cpu_relative",
    "next_obs_memory_relative",
    "next_obs_response_time",
    "terminated",
    "truncated",
    "cpu",
    "memory",
    "response_time",
    "replicas",
}

CURRENT_7D_REQUIRED = {
    "timestamp",
    "episode",
    "step",
    "obs_action",
    "obs_cpu",
    "obs_memory",
    "obs_response_time",
    "obs_cpu_delta",
    "obs_memory_delta",
    "obs_rt_delta",
    "action",
    "reward",
    "next_obs_action",
    "next_obs_cpu",
    "next_obs_memory",
    "next_obs_response_time",
    "next_obs_cpu_delta",
    "next_obs_memory_delta",
    "next_obs_rt_delta",
    "terminated",
    "truncated",
    "cpu",
    "memory",
    "response_time",
    "replicas",
}

LEGACY_MAP = {
    "obs_cpu_relative": "obs_cpu",
    "obs_memory_relative": "obs_memory",
    "next_obs_cpu_relative": "next_obs_cpu",
    "next_obs_memory_relative": "next_obs_memory",
}

# Legacy "relative" columns were generated with calculate_distance():
# relative = (value - min) / (max - min), with min=20 and max=80.
# Source: agent/stable/utils/monitor.py and validated on legacy CSVs.
LEGACY_MIN_CPU = 20.0
LEGACY_MAX_CPU = 80.0
LEGACY_MIN_MEMORY = 20.0
LEGACY_MAX_MEMORY = 80.0

HEADERLESS_4D_COLUMNS = [
    "timestamp",
    "episode",
    "step",
    "obs_action",
    "obs_cpu",
    "obs_memory",
    "obs_response_time",
    "action",
    "reward",
    "next_obs_action",
    "next_obs_cpu",
    "next_obs_memory",
    "next_obs_response_time",
    "terminated",
    "truncated",
    "cpu",
    "memory",
    "response_time",
    "replicas",
]


@dataclass
class FileStats:
    name: str
    schema: str
    rows_total: int = 0
    rows_written: int = 0
    rows_skipped: int = 0


def to_float(value, default: float = 0.0) -> float:
    try:
        num = float(value)
        if math.isnan(num):
            return default
        return num
    except Exception:
        return default


def to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def to_bool_str(value) -> str:
    if isinstance(value, bool):
        return str(value)
    value_str = str(value).strip().lower()
    return str(value_str in {"1", "true", "yes", "y"})


def to_iso_timestamp(value) -> str:
    raw = str(value).strip()
    try:
        datetime.fromisoformat(raw)
        return raw
    except Exception:
        return ""


def has_value(value) -> bool:
    return value is not None and str(value).strip() != ""


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_cpu_memory_from_percent(raw_percent: float) -> float:
    return clip(raw_percent / 100.0, 0.0, 1.0)


def normalize_rt_from_percent(raw_percent: float) -> float:
    return clip(raw_percent / 100.0, 0.0, 3.0)


def legacy_relative_to_percent(
    relative: float,
    min_value: float,
    max_value: float,
) -> float:
    return relative * (max_value - min_value) + min_value


def detect_schema(path: Path) -> str:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        first_row = next(reader, [])

    first_set = set(first_row)
    if CURRENT_7D_REQUIRED.issubset(first_set):
        return "current_7d_header"
    if CURRENT_4D_REQUIRED.issubset(first_set):
        return "current_4d_header"
    if LEGACY_4D_REQUIRED.issubset(first_set):
        return "legacy_4d_header"

    # Headerless files with old 4D format in fixed order.
    if len(first_row) == len(HEADERLESS_4D_COLUMNS):
        first_val = first_row[0].strip()
        try:
            datetime.fromisoformat(first_val)
            return "headerless_4d"
        except Exception:
            pass

    return "unsupported"


def normalize_base_4d_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "timestamp": to_iso_timestamp(row.get("timestamp", "")),
        "episode": str(to_int(row.get("episode", "0"))),
        "step": str(to_int(row.get("step", "0"))),
        "obs_action": str(to_float(row.get("obs_action", "0.0"))),
        "obs_cpu": str(to_float(row.get("obs_cpu", "0.0"))),
        "obs_memory": str(to_float(row.get("obs_memory", "0.0"))),
        "obs_response_time": str(to_float(row.get("obs_response_time", "0.0"))),
        "action": str(to_int(row.get("action", "0"))),
        "reward": str(to_float(row.get("reward", "0.0"))),
        "next_obs_action": str(to_float(row.get("next_obs_action", "0.0"))),
        "next_obs_cpu": str(to_float(row.get("next_obs_cpu", "0.0"))),
        "next_obs_memory": str(to_float(row.get("next_obs_memory", "0.0"))),
        "next_obs_response_time": str(
            to_float(row.get("next_obs_response_time", "0.0"))
        ),
        "terminated": to_bool_str(row.get("terminated", "False")),
        "truncated": to_bool_str(row.get("truncated", "False")),
        "cpu": str(to_float(row.get("cpu", "0.0"))),
        "memory": str(to_float(row.get("memory", "0.0"))),
        "response_time": str(to_float(row.get("response_time", "0.0"))),
        "replicas": str(to_int(row.get("replicas", "0"))),
    }


def normalize_7d_row(row: dict[str, str]) -> dict[str, str]:
    out = {k: "" for k in OUT_COLUMNS}
    for key in OUT_COLUMNS:
        out[key] = row.get(key, "")

    out["timestamp"] = to_iso_timestamp(out["timestamp"])
    out["episode"] = str(to_int(out["episode"], default=0))
    out["step"] = str(to_int(out["step"], default=0))
    out["action"] = str(to_int(out["action"], default=0))
    out["replicas"] = str(to_int(out["replicas"], default=0))
    out["terminated"] = to_bool_str(out["terminated"])
    out["truncated"] = to_bool_str(out["truncated"])

    for key in OUT_COLUMNS:
        if key in {"timestamp", "episode", "step", "action", "replicas", "terminated", "truncated"}:
            continue
        out[key] = str(to_float(out[key], default=0.0))

    return out


def iter_rows(path: Path, schema: str):
    if schema in {"current_4d_header", "legacy_4d_header", "current_7d_header"}:
        with path.open(newline="") as f:
            yield from csv.DictReader(f)
        return

    if schema == "headerless_4d":
        with path.open(newline="") as f:
            reader = csv.reader(f)
            for values in reader:
                if len(values) != len(HEADERLESS_4D_COLUMNS):
                    continue
                yield dict(zip(HEADERLESS_4D_COLUMNS, values, strict=False))
        return

    return


def convert(source_dir: Path, output_path: Path, pattern: str) -> None:
    files = sorted(source_dir.glob(pattern))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats: list[FileStats] = []
    total_written = 0
    total_skipped = 0

    with output_path.open("w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=OUT_COLUMNS)
        writer.writeheader()

        for path in files:
            if path.resolve() == output_path.resolve():
                continue
            if not path.is_file():
                continue

            schema = detect_schema(path)
            file_stats = FileStats(name=path.name, schema=schema)

            if schema == "unsupported":
                file_stats.rows_skipped += 1
                total_skipped += 1
                stats.append(file_stats)
                continue

            # Track previous observation per episode per file only.
            # Episode IDs usually restart from 1 in each CSV file.
            prev_obs_by_episode: dict[int, tuple[float, float, float]] = {}
            prev_next_raw_by_episode: dict[int, tuple[float, float, float]] = {}

            for row in iter_rows(path, schema):
                file_stats.rows_total += 1

                if schema == "current_7d_header":
                    out_row = normalize_7d_row(row)
                    if not out_row["timestamp"]:
                        file_stats.rows_skipped += 1
                        total_skipped += 1
                        continue
                    writer.writerow(out_row)
                    file_stats.rows_written += 1
                    total_written += 1
                    continue

                if schema == "legacy_4d_header":
                    mapped = dict(row)
                    for old_key, new_key in LEGACY_MAP.items():
                        mapped[new_key] = row.get(old_key, "")
                    base = normalize_base_4d_row(mapped)
                    episode = to_int(base["episode"], default=0)

                    # Reconstruct next raw metrics from info fields first.
                    # These are the exact post-action metrics logged by env.
                    if has_value(row.get("cpu")):
                        next_cpu_raw = to_float(row.get("cpu", 0.0), default=0.0)
                    elif has_value(row.get("cpu_relative")):
                        next_cpu_raw = legacy_relative_to_percent(
                            relative=to_float(row.get("cpu_relative", 0.0), default=0.0),
                            min_value=LEGACY_MIN_CPU,
                            max_value=LEGACY_MAX_CPU,
                        )
                    else:
                        next_cpu_raw = legacy_relative_to_percent(
                            relative=to_float(
                                row.get("next_obs_cpu_relative", 0.0),
                                default=0.0,
                            ),
                            min_value=LEGACY_MIN_CPU,
                            max_value=LEGACY_MAX_CPU,
                        )

                    if has_value(row.get("memory")):
                        next_mem_raw = to_float(row.get("memory", 0.0), default=0.0)
                    elif has_value(row.get("memory_relative")):
                        next_mem_raw = legacy_relative_to_percent(
                            relative=to_float(
                                row.get("memory_relative", 0.0),
                                default=0.0,
                            ),
                            min_value=LEGACY_MIN_MEMORY,
                            max_value=LEGACY_MAX_MEMORY,
                        )
                    else:
                        next_mem_raw = legacy_relative_to_percent(
                            relative=to_float(
                                row.get("next_obs_memory_relative", 0.0),
                                default=0.0,
                            ),
                            min_value=LEGACY_MIN_MEMORY,
                            max_value=LEGACY_MAX_MEMORY,
                        )

                    if has_value(row.get("response_time")):
                        next_rt_raw = to_float(
                            row.get("response_time", 0.0),
                            default=0.0,
                        )
                    else:
                        # Legacy next_obs_response_time is already normalized (0-3).
                        next_rt_raw = (
                            to_float(row.get("next_obs_response_time", 0.0), default=0.0)
                            * 100.0
                        )

                    # Reconstruct current raw metrics:
                    # - Prefer previous row's next raw metrics for continuity.
                    # - For first row of episode, invert legacy obs_*_relative.
                    prev_next_raw = prev_next_raw_by_episode.get(episode)
                    if prev_next_raw is not None:
                        obs_cpu_raw, obs_mem_raw, obs_rt_raw = prev_next_raw
                    else:
                        obs_cpu_raw = legacy_relative_to_percent(
                            relative=to_float(
                                row.get("obs_cpu_relative", 0.0),
                                default=0.0,
                            ),
                            min_value=LEGACY_MIN_CPU,
                            max_value=LEGACY_MAX_CPU,
                        )
                        obs_mem_raw = legacy_relative_to_percent(
                            relative=to_float(
                                row.get("obs_memory_relative", 0.0),
                                default=0.0,
                            ),
                            min_value=LEGACY_MIN_MEMORY,
                            max_value=LEGACY_MAX_MEMORY,
                        )
                        obs_rt_raw = (
                            to_float(row.get("obs_response_time", 0.0), default=0.0)
                            * 100.0
                        )

                    base["obs_cpu"] = str(
                        normalize_cpu_memory_from_percent(obs_cpu_raw)
                    )
                    base["obs_memory"] = str(
                        normalize_cpu_memory_from_percent(obs_mem_raw)
                    )
                    base["obs_response_time"] = str(
                        normalize_rt_from_percent(obs_rt_raw)
                    )
                    base["next_obs_cpu"] = str(
                        normalize_cpu_memory_from_percent(next_cpu_raw)
                    )
                    base["next_obs_memory"] = str(
                        normalize_cpu_memory_from_percent(next_mem_raw)
                    )
                    base["next_obs_response_time"] = str(
                        normalize_rt_from_percent(next_rt_raw)
                    )

                    prev_next_raw_by_episode[episode] = (
                        next_cpu_raw,
                        next_mem_raw,
                        next_rt_raw,
                    )
                else:
                    base = normalize_base_4d_row(row)

                if not base["timestamp"]:
                    file_stats.rows_skipped += 1
                    total_skipped += 1
                    continue

                episode = to_int(base["episode"], default=0)
                obs_cpu = to_float(base["obs_cpu"], default=0.0)
                obs_mem = to_float(base["obs_memory"], default=0.0)
                obs_rt = to_float(base["obs_response_time"], default=0.0)

                prev = prev_obs_by_episode.get(episode)
                if prev is None:
                    obs_cpu_delta = 0.0
                    obs_mem_delta = 0.0
                    obs_rt_delta = 0.0
                else:
                    obs_cpu_delta = obs_cpu - prev[0]
                    obs_mem_delta = obs_mem - prev[1]
                    obs_rt_delta = obs_rt - prev[2]

                next_obs_cpu = to_float(base["next_obs_cpu"], default=0.0)
                next_obs_mem = to_float(base["next_obs_memory"], default=0.0)
                next_obs_rt = to_float(base["next_obs_response_time"], default=0.0)

                next_cpu_delta = next_obs_cpu - obs_cpu
                next_mem_delta = next_obs_mem - obs_mem
                next_rt_delta = next_obs_rt - obs_rt

                out_row = {
                    "timestamp": base["timestamp"],
                    "episode": base["episode"],
                    "step": base["step"],
                    "obs_action": base["obs_action"],
                    "obs_cpu": base["obs_cpu"],
                    "obs_memory": base["obs_memory"],
                    "obs_response_time": base["obs_response_time"],
                    "obs_cpu_delta": str(obs_cpu_delta),
                    "obs_memory_delta": str(obs_mem_delta),
                    "obs_rt_delta": str(obs_rt_delta),
                    "action": base["action"],
                    "reward": base["reward"],
                    "next_obs_action": base["next_obs_action"],
                    "next_obs_cpu": base["next_obs_cpu"],
                    "next_obs_memory": base["next_obs_memory"],
                    "next_obs_response_time": base["next_obs_response_time"],
                    "next_obs_cpu_delta": str(next_cpu_delta),
                    "next_obs_memory_delta": str(next_mem_delta),
                    "next_obs_rt_delta": str(next_rt_delta),
                    "terminated": base["terminated"],
                    "truncated": base["truncated"],
                    "cpu": base["cpu"],
                    "memory": base["memory"],
                    "response_time": base["response_time"],
                    "replicas": base["replicas"],
                }

                writer.writerow(out_row)
                file_stats.rows_written += 1
                total_written += 1
                prev_obs_by_episode[episode] = (obs_cpu, obs_mem, obs_rt)

            stats.append(file_stats)

    print(f"Source directory : {source_dir}")
    print(f"Output file      : {output_path}")
    print(f"Files scanned    : {len(files)}")
    print(f"Rows written     : {total_written}")
    print(f"Rows skipped     : {total_skipped}")
    print("")
    print("Per-file stats:")
    for s in stats:
        print(
            f"- {s.name}: schema={s.schema}, total={s.rows_total}, "
            f"written={s.rows_written}, skipped={s.rows_skipped}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert agent/stable CSV transitions to stable_simple 7D format."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("agent/stable/data"),
        help="Directory containing source CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("agent/stable_simple/data/from_stable_converted.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for input files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        source_dir=args.source_dir,
        output_path=args.output,
        pattern=args.pattern,
    )
