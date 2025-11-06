import logging
import time

import numpy as np
from prometheus_api_client import PrometheusApiClientException, PrometheusConnect


def _metrics_query(
    namespace: str,
    deployment_name: str,
    interval: int = 15,
) -> tuple[str, str, str, str]:
    cpu_query = f"""
        (
            sum by (pod) (
                rate(container_cpu_usage_seconds_total{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    container!="",
                    container!="POD"
                }}[{interval}s])
            )
        ) OR (
            count by (pod) (
                container_cpu_usage_seconds_total{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    container!="",
                    container!="POD"
                }}
            ) * 0
        )
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
        """

    return (
        cpu_query,
        memory_query,
        cpu_limits_query,
        memory_limits_query,
    )


def _metrics_result(
    cpu_limits_results: list,
    memory_limits_results: list,
    cpu_usage_results: list,
    memory_usage_results: list,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
) -> tuple[list[float], list[float], set[str]]:
    cpu_percentages = []
    memory_percentages = []
    pod_names = set()

    cpu_limits_by_pod = {}
    memory_limits_by_pod = {}

    for result in cpu_limits_results:
        pod_name = result["metric"].get("pod")
        if pod_name:
            cpu_limits_by_pod[pod_name] = float(result["value"][1])

    for result in memory_limits_results:
        pod_name = result["metric"].get("pod")
        if pod_name:
            memory_limits_by_pod[pod_name] = float(result["value"][1])

    for result in cpu_usage_results:
        pod_name = result["metric"].get("pod")
        if not pod_name or pod_name not in cpu_limits_by_pod:
            logger.warning(f"Skipping pod {pod_name}: CPU limit missing")
            continue

        rate_cores = float(result["value"][1])  # cores
        limit_cores = float(cpu_limits_by_pod[pod_name])  # cores

        if limit_cores <= 0:
            logger.warning(f"CPU limit not set or zero for pod {pod_name}")
            continue

        cpu_percentage = (rate_cores / limit_cores) * 100
        cpu_percentages.append(cpu_percentage)
        pod_names.add(pod_name)
        logger.debug(
            f"Pod {pod_name}: CPU {rate_cores:.4f} cores / "
            f"{limit_cores} -> {cpu_percentage:.2f}%"
        )

    for result in memory_usage_results:
        pod_name = result["metric"].get("pod")
        if (
            not pod_name
            or pod_name not in memory_limits_by_pod
            or pod_name not in pod_names
        ):
            logger.warning(f"Skipping pod {pod_name}: Memory limit missing")
            continue

        used_bytes = float(result["value"][1])
        limit_bytes = float(memory_limits_by_pod[pod_name])
        if limit_bytes <= 0:
            logger.warning(f"Memory limit not set or zero for pod {pod_name}")
            continue

        memory_percentage = (used_bytes / limit_bytes) * 100.0
        logger.debug(
            f"Pod {pod_name}: Memory {used_bytes:.2f} bytes / "
            f"{limit_bytes} -> {memory_percentage:.2f}%"
        )
        memory_percentages.append(memory_percentage)
        pod_names.add(pod_name)

        logger.debug(f"Pod names with metrics: {pod_names}")
        logger.debug(f"CPU percentages: {cpu_percentages}")
        logger.debug(f"Memory percentages: {memory_percentages}")
    return cpu_percentages, memory_percentages, pod_names


def _get_response_time(
    prometheus: PrometheusConnect,
    deployment_name: str,
    namespace: str = "default",
    endpoints_method: list[tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
    interval: int = 15,
    quantile: float = 0.90,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
) -> float:
    result = []
    for endpoint, method in endpoints_method:
        q = f"""
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

        """
        try:
            response = prometheus.custom_query(q)
            if response:
                for res in response:
                    if "value" in res and len(res["value"]) > 1:
                        result.append(float(res["value"][1]))
                    else:
                        result.append(0.0)
            else:
                response = 0.0
                result.append(response)
        except PrometheusApiClientException as e:
            if "404 page not found" in str(e):
                logger.warning(
                    f"Prometheus custom query returned 404 for app={deployment_name}, "
                    f"namespace={namespace}, endpoint={endpoint}, "
                    f"method={method}. Error: {e}"
                )
                result.append(0.0)
            else:
                logger.error(
                    f"Prometheus custom query failed for app={deployment_name}, "
                    f"namespace={namespace}, endpoint={endpoint}, "
                    f"method={method}. Error: {e}"
                )
                result.append(0.0)
        except Exception as e:
            logger.error(
                f"Prometheus custom query failed for app={deployment_name}, "
                f"namespace={namespace}, endpoint={endpoint}, "
                f"method={method}. Error: {e}"
            )
            result.append(0.0)

    # Convert to numpy array and filter out non-finite values to avoid
    # RuntimeWarning from np.nanmean when the array is all-NaN.
    arr = np.array(result, dtype=float) if result else np.array([], dtype=float)
    finite = arr[np.isfinite(arr)]

    if finite.size > 0:
        response_time = float(np.mean(finite))
    else:
        logger.warning(
            f"Response time calculation resulted in NaN for endpoints "
            f"{endpoints_method}. This typically means no request data "
            f"is available in the {interval}s time window. Defaulting to 0.0 ms."
        )
        response_time = 0.0

    logger.debug(
        f"Response time (quantile {quantile}) for endpoints "
        f"{endpoints_method}: {response_time} ms"
    )

    return response_time


def _scrape_metrics(
    fetch_timeout: int,
    prometheus: PrometheusConnect,
    cpu_query: str,
    memory_query: str,
    cpu_limits_query: str,
    memory_limits_query: str,
    replicas: int,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
) -> tuple[list, list, list, list]:
    fetch_start = time.time()

    while time.time() - fetch_start < fetch_timeout:
        cpu_usage_results = prometheus.custom_query(cpu_query)
        logger.debug(f"Fetched {len(cpu_usage_results)} CPU usage entries")
        if len(cpu_usage_results) != replicas:
            logger.debug(
                f"Expected {replicas} CPU usage results, got {len(cpu_usage_results)}"
            )
            time.sleep(1)
            continue
        break

    fetch_start = time.time()
    while time.time() - fetch_start < fetch_timeout:
        memory_usage_results = prometheus.custom_query(memory_query)
        logger.debug(f"Fetched {len(memory_usage_results)} Memory usage entries")
        if len(memory_usage_results) != replicas:
            logger.debug(
                f"Expected {replicas} Memory usage results, got "
                f"{len(memory_usage_results)}"
            )
            time.sleep(1)
            continue
        break

    fetch_start = time.time()
    while time.time() - fetch_start < fetch_timeout:
        cpu_limits_results = prometheus.custom_query(cpu_limits_query)
        logger.debug(f"Fetched {len(cpu_limits_results)} CPU limits entries")
        if len(cpu_limits_results) != replicas:
            logger.debug(
                f"Expected {replicas} CPU limits results, got {len(cpu_limits_results)}"
            )
            time.sleep(1)
            continue
        break

    fetch_start = time.time()
    while time.time() - fetch_start < fetch_timeout:
        memory_limits_results = prometheus.custom_query(memory_limits_query)
        logger.debug(f"Fetched {len(memory_limits_results)} Memory limits entries")
        if len(memory_limits_results) != replicas:
            logger.debug(
                f"Expected {replicas} Memory limits results, "
                f"got {len(memory_limits_results)}"
            )
            time.sleep(1)
            continue
        break

    logger.debug(
        f"Fetched metrics: CPU usage {len(cpu_usage_results)} entries, "
        f"Memory usage {len(memory_usage_results)} entries"
    )
    logger.debug(
        f"Fetched metrics: CPU limits {len(cpu_limits_results)} entries, "
        f"Memory limits {len(memory_limits_results)} entries"
    )

    return (
        cpu_usage_results,
        memory_usage_results,
        cpu_limits_results,
        memory_limits_results,
    )


def get_metrics(
    replicas: int,
    timeout: int,
    namespace: str,
    deployment_name: str,
    wait_time: int,
    prometheus: PrometheusConnect,
    interval: int = 15,
    quantile: float = 0.90,
    endpoints_method: list[tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
    increase: bool = False,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
) -> tuple[float, float, float, int]:
    """
    Returns (cpu_usage_mean, memory_usage_mean, response_time, replica_count)
    cpu in %, memory in %, averaged over matched pods.
    """
    if increase or wait_time > 0:
        time.sleep(wait_time)

    start = time.time()
    while time.time() - start < timeout:
        try:
            prometheus.check_prometheus_connection()
        except Exception as e:
            logger.warning(f"Prometheus connectivity issue: {e}")
            time.sleep(1)
            continue

        cpu_query, memory_query, cpu_limits_query, memory_limits_query = _metrics_query(
            namespace, deployment_name, interval=interval
        )
        logger.debug("Metrics queries prepared, querying Prometheus...")
        logger.debug(f"CPU Query: {cpu_query}")
        logger.debug(f"Memory Query: {memory_query}")
        logger.debug(f"CPU Limits Query: {cpu_limits_query}")
        logger.debug(f"Memory Limits Query: {memory_limits_query}")

        try:
            fetch_timeout = timeout / 2

            (
                cpu_usage_results,
                memory_usage_results,
                cpu_limits_results,
                memory_limits_results,
            ) = _scrape_metrics(
                fetch_timeout=fetch_timeout,
                prometheus=prometheus,
                cpu_query=cpu_query,
                memory_query=memory_query,
                cpu_limits_query=cpu_limits_query,
                memory_limits_query=memory_limits_query,
                replicas=replicas,
                logger=logger,
            )
            if not cpu_usage_results or not memory_usage_results:
                logger.debug("No metrics found, retrying...")
                time.sleep(1)
                continue

            logger.debug(
                f"metrics found: CPU usage {len(cpu_usage_results)} entries, "
                f"Memory usage {len(memory_usage_results)} entries"
            )

            cpu_percentages, memory_percentages, pod_names = _metrics_result(
                cpu_limits_results,
                memory_limits_results,
                cpu_usage_results,
                memory_usage_results,
                logger=logger,
            )

            response_time = _get_response_time(
                prometheus=prometheus,
                deployment_name=deployment_name,
                namespace=namespace,
                endpoints_method=endpoints_method,
                interval=interval,
                quantile=quantile,
                logger=logger,
            )

            collected = len(pod_names)
            if collected == 0:
                logger.warning(
                    "No eligible pods (limits missing or no Ready pods). Retrying..."
                )
                time.sleep(1)
                continue

            if collected == replicas:
                cpu_mean = float(np.mean(cpu_percentages)) if cpu_percentages else 0.0
                mem_mean = (
                    float(np.mean(memory_percentages)) if memory_percentages else 0.0
                )

                if not cpu_percentages:
                    logger.warning("No valid CPU percentages calculated.")
                if not memory_percentages:
                    logger.warning("No valid Memory percentages calculated.")

                # Verify data consistency
                if len(cpu_percentages) != len(memory_percentages):
                    logger.error(
                        f"Mismatch: {len(cpu_percentages)} CPU entries vs "
                        f"{len(memory_percentages)} Memory entries"
                    )

                logger.debug(
                    f"Metrics collected from {collected} pods: \n"
                    f"CPU usage mean {cpu_mean:.3f}%, \n"
                    f"Memory usage mean {mem_mean:.3f}%, \n"
                    f"Response time {response_time:.3f} ms"
                )
                return cpu_mean, mem_mean, response_time, collected
            logger.warning(
                f"Only collected metrics from {collected} pods, expected {replicas}"
            )
            continue

        except PrometheusApiClientException as e:
            logger.error(f"Prometheus query failed: {e}")
        except Exception as e:
            logger.error(f"Error processing Prometheus metrics: {e}")

        time.sleep(1)

    logger.error("Timeout reached while fetching metrics.")
    return 0.0, 0.0, 0.0, 0
