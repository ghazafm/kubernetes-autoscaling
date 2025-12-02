import numpy as np
from prometheus_api_client import PrometheusConnect


def _metrics_query(
    namespace: str,
    deployment_name: str,
    interval: int = 15,
    desired_replicas: int | None = None,
    quantile: float = 0.90,
    endpoints_method: list[tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
) -> tuple[str, str, str, str, str]:
    """
    Build pod-scoped queries and cap to the youngest desired pods.

    We use topk on pod start time to keep only the newest N pods (desired replicas),
    so older pods that are still Ready after a scale-down do not contribute.
    """
    # Default to a reasonable cap if desired_replicas is None
    pod_window = max(1, desired_replicas or 50)

    pod_filter = f"""
        topk({pod_window},
          kube_pod_start_time{{
            namespace="{namespace}",
            pod=~"{deployment_name}-.*"
          }}
          * on(pod) group_left()
            (kube_pod_status_ready{{
                namespace="{namespace}",
                pod=~"{deployment_name}-.*",
                condition="true"
            }} == 1)
        )
    """

    cpu_query = f"""
        sum by (pod) (
            rate(container_cpu_usage_seconds_total{{
                namespace="{namespace}",
                pod=~"{deployment_name}-.*",
                container!="",
                container!="POD"
            }}[{interval}s])
        )
        * on(pod) group_left() {pod_filter}
        """

    memory_query = f"""
        sum by (pod) (
            container_memory_working_set_bytes{{
                namespace="{namespace}",
                pod=~"{deployment_name}-.*",
                container!="",
                container!="POD"
            }}
        )
        * on(pod) group_left() {pod_filter}
        """

    cpu_limits_query = f"""
        sum by (pod) (
            kube_pod_container_resource_limits{{
                namespace="{namespace}",
                pod=~"{deployment_name}-.*",
                resource="cpu",
                unit="core"
            }}
        )
        * on(pod) group_left() {pod_filter}
        """

    # Query for memory limits
    memory_limits_query = f"""
        sum by (pod) (
            kube_pod_container_resource_limits{{
                namespace="{namespace}",
                pod=~"{deployment_name}-.*",
                resource="memory",
                unit="byte"
            }}
        )
        * on(pod) group_left() {pod_filter}
        """

    response_time_query = []
    for endpoint, method in endpoints_method:
        response_time_query.append(f"""
                1000 *
                histogram_quantile(
                {quantile},
                sum by (le) (
                    rate(http_request_duration_seconds_bucket{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    method="{method}",
                    path="{endpoint}"
                    }}[{interval}s])
                )
                )
            """)
    return (
        cpu_query,
        memory_query,
        cpu_limits_query,
        memory_limits_query,
        response_time_query,
    )


def process_metrics(
    cpu_usage,
    memory_usage,
    cpu_limits,
    memory_limits,
    response_times,
    max_response_time,
):
    cpu_percentages = []
    memory_percentages = []

    cpu_limits_by_pod = {}
    memory_limits_by_pod = {}
    for item in cpu_limits:
        pod = item["metric"].get("pod")
        if pod:
            limit = float(item["value"][1])
            cpu_limits_by_pod[pod] = limit

    for item in memory_limits:
        pod = item["metric"].get("pod")
        if pod:
            limit = float(item["value"][1])
            memory_limits_by_pod[pod] = limit

    for result in cpu_usage:
        pod_name = result["metric"].get("pod")
        limit = cpu_limits_by_pod.get(pod_name)
        if limit is None or limit == 0:
            continue  # Skip pods without limits
        rate_cores = float(result["value"][1])
        cpu_percentage = (rate_cores / limit) * 100
        cpu_percentages.append(cpu_percentage)

    for result in memory_usage:
        pod_name = result["metric"].get("pod")
        limit = memory_limits_by_pod.get(pod_name)
        if limit is None or limit == 0:
            continue  # Skip pods without limits
        usage_bytes = float(result["value"][1])
        memory_percentage = (usage_bytes / limit) * 100
        memory_percentages.append(memory_percentage)

    # Handle empty arrays to avoid nan from np.mean
    response_time = np.mean(response_times) if response_times else 0.0
    response_time_percentage = (response_time / max_response_time) * 100.0
    response_time_percentage = min(response_time_percentage, 1000.0)

    # Return 0.0 for empty arrays instead of nan
    cpu_mean = float(np.mean(cpu_percentages)) if cpu_percentages else 0.0
    memory_mean = float(np.mean(memory_percentages)) if memory_percentages else 0.0

    return (
        cpu_mean,
        memory_mean,
        response_time_percentage,
    )


def get_metrics(
    prometheus: PrometheusConnect,
    namespace: str,
    deployment_name: str,
    interval: int,
    replica: int,
    max_response_time: float,
    quantile: float = 0.90,
    endpoints_method: list[tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
):
    (
        cpu_query,
        memory_query,
        cpu_limits_query,
        memory_limits_query,
        response_time_query,
    ) = _metrics_query(
        namespace,
        deployment_name,
        interval=interval,
        desired_replicas=replica,
        quantile=quantile,
        endpoints_method=endpoints_method,
    )
    cpu_usage_results = prometheus.custom_query(cpu_query)
    memory_usage_results = prometheus.custom_query(memory_query)
    cpu_limits_results = prometheus.custom_query(cpu_limits_query)
    memory_limits_results = prometheus.custom_query(memory_limits_query)

    response_time_results = []
    for query in response_time_query:
        response = prometheus.custom_query(query)
        if not response:
            response_time_results.append(0.0)
            continue

        response_time_results.append(float(response[0]["value"][1]))

    cpu_percentages, memory_percentages, response_time_percentage = process_metrics(
        cpu_usage_results,
        memory_usage_results,
        cpu_limits_results,
        memory_limits_results,
        response_time_results,
        max_response_time,
    )

    return cpu_percentages, memory_percentages, response_time_percentage
