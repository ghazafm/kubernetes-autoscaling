#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

INFLUX_URL = os.environ["INFLUXDB_URL"].rstrip("/")
INFLUX_ORG = os.environ["INFLUXDB_ORG"]
INFLUX_TOKEN = os.environ["INFLUXDB_TOKEN"]
BUCKET = os.environ.get("INFLUXDB_BUCKET")
WINDOWS_FILE = os.environ.get("WINDOWS_FILE", "time_windows.csv")
WINDOW_EVERY = os.environ.get("WINDOW_EVERY", "10s")
DEPLOYMENT_FILTER = (
    'r["deployment"] == "test-flask-app" or r["deployment"] == "hpa-flask-app"'
)
WIB = timezone(timedelta(hours=7))


def make_query(field: str, aggregate_fn: str, yield_name: str, namespace: str) -> str:
    namespace_filter = (
        f'  |> filter(fn: (r) => r["namespace"] == "{namespace}")\n' if namespace else ""
    )
    return f"""
from(bucket: "{BUCKET}")
  |> range(start: time(v: "{{start}}"), stop: time(v: "{{stop}}"))
  |> filter(fn: (r) => r["_measurement"] == "monitoring_cluster")
  |> filter(fn: (r) => {DEPLOYMENT_FILTER})
{namespace_filter}  |> filter(fn: (r) => r["_field"] == "{field}")
  |> aggregateWindow(every: {WINDOW_EVERY}, fn: {aggregate_fn}, createEmpty: false)
  |> yield(name: "{yield_name}")
""".strip()


QUERY_BY_METRIC = {
    "response_time": make_query(
        field="response_time_raw",
        aggregate_fn="mean",
        yield_name="mean",
        namespace="default",
    ),
    "replica": make_query(
        field="desired_replicas",
        aggregate_fn="last",
        yield_name="last",
        namespace="",
    ),
    "memory": make_query(
        field="memory",
        aggregate_fn="mean",
        yield_name="mean",
        namespace="default",
    ),
    "cpu": make_query(
        field="cpu",
        aggregate_fn="mean",
        yield_name="mean",
        namespace="default",
    ),
}


def influx_query_csv(flux_query: str) -> str:
    endpoint = f"{INFLUX_URL}/api/v2/query?org={urllib.parse.quote(INFLUX_ORG)}"
    request = urllib.request.Request(
        endpoint,
        data=flux_query.encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Token {INFLUX_TOKEN}",
            "Accept": "application/csv",
            "Content-Type": "application/vnd.flux",
        },
    )
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8")


def count_data_rows(csv_text: str) -> int:
    rows = 0
    for line in csv_text.splitlines():
        if not line or line.startswith("#") or line.startswith(",result,"):
            continue
        rows += 1
    return rows


def to_influx_utc_time(time_value: str) -> str:
    value = time_value.strip()

    if value.endswith("Z"):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif "T" in value:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=WIB)
    else:
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=WIB)

    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def iter_windows(path: str):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("dataset"):
                continue
            yield row["dataset"], row["run"], row["start"], row["stop"]


def main() -> None:
    for dataset, run, start, stop in iter_windows(WINDOWS_FILE):
        start_utc = to_influx_utc_time(start)
        stop_utc = to_influx_utc_time(stop)
        for metric, query_template in QUERY_BY_METRIC.items():
            query = query_template.format(start=start_utc, stop=stop_utc)
            csv_text = influx_query_csv(query)

            output_path = Path("data") / dataset / metric / f"{run}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(csv_text, encoding="utf-8")

            row_count = count_data_rows(csv_text)
            print(f"{output_path}: {row_count} rows")


if __name__ == "__main__":
    main()
