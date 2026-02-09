# Test Analysis Directory

This directory contains CSV data exports from k6 load tests and analysis notebooks.

## Files

### Data Files

- **`request.csv`** - Aggregate request counts by endpoint type (basic, cpu, memory)
- **`10_request.csv`** - Time-series request data from 10-replica test
- **`20_request.csv`** - Time-series request data from 20-replica test
- **`10_pod.csv`** - Pod scaling metrics from 10-replica test
- **`20_pod.csv`** - Pod scaling metrics from 20-replica test
- **`request_timeseries.csv`** - Full time-series request data
- **`cpu.csv`**, **`memory.csv`**, **`rt.csv`**, etc. - Various metrics CSVs

### Analysis Files

- **`notebook.ipynb`** - Main analysis notebook for request.csv
- **`notebook_update.ipynb`** - Updated analysis with visualizations for 10_request.csv

### Scripts

- **`generate_request_csv.py`** - Automated script to query InfluxDB and generate request.csv
- **`create_balanced_data.py`** - Create synthetic 50/50 balanced request distribution
- **`observation_demo.py`** - Demo script (in parent test/ directory)

## Data Source

All CSV files are exported from **InfluxDB v2**:
- **URL:** `http://10.34.4.192:8086`
- **Organization:** `icn`
- **Bucket:** `autoscaling-reinforcement-learning`

The data comes from **k6 load tests** that send metrics directly to InfluxDB.

## Quick Start

### Generate request.csv from InfluxDB

```bash
# Set your InfluxDB token
export INFLUXDB_TOKEN="your-token-here"

# Generate aggregate request counts (last 24 hours)
python generate_request_csv.py

# Generate from specific time range
python generate_request_csv.py --start "-1h"

# Generate time-series data
python generate_request_csv.py --timeseries --start "-1h" --output my_timeseries.csv
```

### Create Balanced Data for Thesis

```bash
# Create 50/50 balanced distribution from existing data
python create_balanced_data.py

# Output: 10_request_balanced.csv
```

### Analyze Data

```bash
# Open Jupyter notebook
jupyter notebook notebook_update.ipynb
```

## Data Structure

### request.csv (Aggregate)

```csv
#group,false,false,false,true
#datatype,string,long,double,string
#default,_result,,,
,result,table,_value,type
,,0,20455,basic
,,1,27264,cpu
,,2,24333,memory
```

### 10_request.csv (Time-Series)

```csv
#group,false,false,true,true,false,false,true,true,true...
#datatype,string,long,dateTime:RFC3339,dateTime:RFC3339,dateTime:RFC3339,double,string...
#default,mean,,,,,,,,,,,,,,,
,result,table,_start,_stop,_time,_value,_field,_measurement,endpoint,expected_response,method,name,proto,scenario,status
,,1,2026-02-06T01:17:00Z,2026-02-06T02:17:00Z,2026-02-06T01:17:10Z,1,value,http_reqs,cpu,true,GET,http://10.34.4.197/api/cpu?iterations=400000,HTTP/1.1,default,200
...
```

## Endpoint Types

Data is categorized by three endpoint types:

1. **`basic`** - Requests to `/api` (no intensive workload)
2. **`cpu`** - Requests to `/api/cpu?iterations=400000` (CPU-intensive)
3. **`memory`** - Requests to `/api/memory?size_mb=30` (memory-intensive)

## Documentation

For detailed information on data generation, see:
- **[HOW_TO_GENERATE_REQUEST_CSV.md](HOW_TO_GENERATE_REQUEST_CSV.md)** - Complete guide to querying InfluxDB

## Common Tasks

### Check Request Distribution

```python
import pandas as pd

df = pd.read_csv("request.csv", comment="#")
print(df)
print("\nTotal:", df["_value"].sum())
print("\nDistribution:")
print(df.groupby("type")["_value"].sum() / df["_value"].sum() * 100)
```

### Visualize Time-Series

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("10_request.csv", comment="#")
df = df[df["_time"] != "_time"].copy()  # Remove duplicate headers
df["_time"] = pd.to_datetime(df["_time"])
df["_value"] = df["_value"].astype(float)

# Plot requests over time by endpoint
for endpoint in df["endpoint"].unique():
    subset = df[df["endpoint"] == endpoint]
    plt.plot(subset["_time"], subset["_value"], label=endpoint)

plt.legend()
plt.xlabel("Time")
plt.ylabel("Requests")
plt.title("HTTP Requests Over Time")
plt.show()
```

### Export from InfluxDB UI

1. Open InfluxDB UI: `http://10.34.4.192:8086`
2. Navigate to **Data Explorer**
3. Click **Script Editor**
4. Paste Flux query (see HOW_TO_GENERATE_REQUEST_CSV.md)
5. Adjust time range
6. Click **Submit**
7. Click **Download CSV**

## References

- [test/README.md](../README.md) - k6 test documentation
- [InfluxDB Flux language](https://docs.influxdata.com/flux/v0/)
- [k6 InfluxDB output](https://grafana.com/docs/k6/latest/results-output/real-time/influxdb/)
