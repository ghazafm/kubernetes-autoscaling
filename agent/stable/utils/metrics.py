import numpy as np
from prometheus_api_client import PrometheusConnect


def _metrics_query(
    namespace: str,
    deployment_name: str,
    interval: int = 15,
    quantile: float = 0.90,
    endpoints_method: list[tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
) -> tuple[str, str, str, str, str]:
    ready_filter = f"""
        kube_pod_status_ready{{
            namespace="{namespace}",
            pod=~"{deployment_name}-.*",
            condition="true"
        }} == 1
    """

    cpu_query = f"""
        quantile({quantile},
            sum by (pod) (
                rate(container_cpu_usage_seconds_total{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    container!="",
                    container!="POD"
                }}[{interval}s])
            )
            * on(pod) group_left() ({ready_filter})
        )
        """

    memory_query = f"""
        quantile({quantile},
            sum by (pod) (
                container_memory_working_set_bytes{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    container!="",
                    container!="POD"
                }}
            )
            * on(pod) group_left() ({ready_filter})
        )
        """

    cpu_limits_query = f"""
        quantile({quantile},
            sum by (pod) (
                kube_pod_container_resource_limits{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    resource="cpu",
                    unit="core"
                }}
            )
            * on(pod) group_left() ({ready_filter})
        )
        """

    memory_limits_query = f"""
        quantile({quantile},
            sum by (pod) (
                kube_pod_container_resource_limits{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    resource="memory",
                    unit="byte"
                }}
            )
            * on(pod) group_left() ({ready_filter})
        )
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
    cpu_value = float(cpu_usage[0]["value"][1]) if cpu_usage else 0.0
    memory_value = float(memory_usage[0]["value"][1]) if memory_usage else 0.0
    cpu_limit = float(cpu_limits[0]["value"][1]) if cpu_limits else 0.0
    memory_limit = float(memory_limits[0]["value"][1]) if memory_limits else 0.0

    if np.isnan(cpu_value):
        cpu_value = 0.0
    if np.isnan(memory_value):
        memory_value = 0.0

    cpu_percentage = (cpu_value / cpu_limit * 100) if cpu_limit > 0 else 0.0
    memory_percentage = (memory_value / memory_limit * 100) if memory_limit > 0 else 0.0

    response_time = np.mean(response_times) if response_times else 0.0
    if np.isnan(response_time):
        response_time = 0.0
    response_time_percentage = (response_time / max_response_time) * 100.0
    response_time_percentage = min(response_time_percentage, 300.0)

    return (
        cpu_percentage,
        memory_percentage,
        response_time_percentage,
    )


def get_metrics(
    prometheus: PrometheusConnect,
    namespace: str,
    deployment_name: str,
    interval: int,
    max_response_time: float,
    quantile: float,
    endpoints_method: list[tuple[str, str]],
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
            continue

        value = float(response[0]["value"][1])
        if not np.isnan(value):
            response_time_results.append(value)

    (
        cpu_percentages,
        memory_percentages,
        response_time_percentage,
    ) = process_metrics(
        cpu_usage_results,
        memory_usage_results,
        cpu_limits_results,
        memory_limits_results,
        response_time_results,
        max_response_time,
    )

    return (
        cpu_percentages,
        memory_percentages,
        response_time_percentage,
    )
