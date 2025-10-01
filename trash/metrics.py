import logging
import time

import numpy as np
from kubernetes.client.api import CoreV1Api, CustomObjectsApi
from prometheus_api_client import PrometheusApiClientException, PrometheusConnect

from .helper import parse_cpu_value, parse_memory_value


def fetch_metrics(api, namespace):
    try:
        return api.list_namespaced_custom_object(
            group="metrics.k8s.io",
            version="v1beta1",
            namespace=namespace,
            plural="pods",
        )
    except Exception as e:
        logging.warning(f"Error fetching metrics: {e}")
        return None


def filter_target_pods(metric_data, deployment_name):
    return [
        item
        for item in metric_data.get("items", [])
        if deployment_name in item["metadata"]["name"]
    ]


def fetch_pod_specs(core, namespace, target_names):
    try:
        pods = core.list_namespaced_pod(namespace=namespace)
        return {
            p.metadata.name: p
            for p in pods.items
            if p.metadata and p.metadata.name in target_names
        }
    except Exception as e:
        logging.warning(f"Error listing pods: {e}")
        return {}


def calculate_usage(item, pod_obj):
    pod_cpu_used_cores = 0.0
    pod_mem_used_bytes = 0.0
    for c in item.get("containers", []):
        try:
            cpu_str = c["usage"]["cpu"]
            mem_str = c["usage"]["memory"]
            pod_cpu_used_cores += parse_cpu_value(cpu_str)
            pod_mem_used_bytes += parse_memory_value(mem_str)
        except Exception as e:
            logging.debug(f"Bad usage entry in {item['metadata']['name']}: {e}")

    pod_cpu_limit_cores = 0.0
    pod_mem_limit_bytes = 0.0
    for c in getattr(pod_obj.spec, "containers", []) or []:
        limits = getattr(c.resources, "limits", {}) or {}
        if "cpu" in limits:
            try:
                pod_cpu_limit_cores += parse_cpu_value(limits["cpu"])
            except Exception as e:
                logging.debug(f"Bad CPU limit in {item['metadata']['name']}: {e}")
        if "memory" in limits:
            try:
                pod_mem_limit_bytes += parse_memory_value(limits["memory"])
            except Exception as e:
                logging.debug(f"Bad memory limit in {item['metadata']['name']}: {e}")

    cpu_pct = (
        (pod_cpu_used_cores / pod_cpu_limit_cores * 100.0)
        if pod_cpu_limit_cores > 0
        else None
    )
    mem_pct = (
        (pod_mem_used_bytes / pod_mem_limit_bytes * 100.0)
        if pod_mem_limit_bytes > 0
        else None
    )
    if cpu_pct is None:
        logging.warning(
            f"CPU limit not set for pod {item['metadata']['name']}; CPU% undefined"
        )
    if mem_pct is None:
        logging.warning(
            f"Memory limit not set for pod {item['metadata']['name']}; "
            f"Memory% undefined"
        )
    return cpu_pct if cpu_pct is not None else float(
        "nan"
    ), mem_pct if mem_pct is not None else float("nan")


def get_metrics(
    replicas: int,
    timeout: int,
    namespace: str,
    deployment_name: str,
    wait_time: int,
    api: CustomObjectsApi,
    core: CoreV1Api,
) -> tuple[float, float, int]:
    """
    Returns (cpu_usage_mean, memory_usage_mean, replica_count)
    cpu in %, memory in %, averaged over matched pods.
    """
    if wait_time > 0:
        time.sleep(wait_time)

    start = time.time()
    while time.time() - start < timeout:
        metric_data = fetch_metrics(api, namespace)
        if not metric_data:
            time.sleep(1)
            continue

        target_metric_items = filter_target_pods(metric_data, deployment_name)
        if not target_metric_items:
            time.sleep(1)
            continue

        target_names = {it["metadata"]["name"] for it in target_metric_items}
        pod_spec_by_name = fetch_pod_specs(core, namespace, target_names)
        if not pod_spec_by_name:
            time.sleep(1)
            continue

        cpu_vals, mem_vals = [], []
        collected = 0

        for item in target_metric_items:
            pod_name = item["metadata"]["name"]
            pod_obj = pod_spec_by_name.get(pod_name)
            if not pod_obj:
                logging.debug(f"Pod spec not found for {pod_name}, skipping")
                continue

            cpu_pct, mem_pct = calculate_usage(item, pod_obj)
            cpu_vals.append(cpu_pct)
            mem_vals.append(mem_pct)
            collected += 1
            if collected >= replicas:
                break

        if collected >= replicas:
            cpu_mean = float(np.nanmean(cpu_vals)) if cpu_vals else 0.0
            mem_mean = float(np.nanmean(mem_vals)) if mem_vals else 0.0
            return cpu_mean, mem_mean, collected

        time.sleep(1)

    return 0.0, 0.0, 0


def get_response_time(
    prometheus: PrometheusConnect,
    app: str,
    namespace: str = "default",
    endpoints_method: list[tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
    interval: int = 15,
    quantile: float = 0.90,
) -> float:
    result = []
    time.sleep(10)
    for endpoint, method in endpoints_method:
        q = f"""
            1000 *
            histogram_quantile(
            {quantile},
            sum by (le) (
                rate(app_request_latency_seconds_bucket{{
                job="{app}",
                namespace="{namespace}",
                method="{method}",
                exported_endpoint="{endpoint}"
                }}[{interval}s])
            )
            )

        """
        try:
            prometheus.check_prometheus_connection()
        except Exception as e:
            logging.warning(f"Prometheus connectivity issue: {e}")
            return 0.0
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
                logging.warning(
                    f"Prometheus custom query returned 404 for app={app}, "
                    f"namespace={namespace}, endpoint={endpoint}, "
                    f"method={method}. Error: {e}"
                )
                result.append(0.0)
            else:
                logging.error(
                    f"Prometheus custom query failed for app={app}, "
                    f"namespace={namespace}, endpoint={endpoint}, "
                    f"method={method}. Error: {e}"
                )
                result.append(0.0)
        except Exception as e:
            logging.error(
                f"Prometheus custom query failed for app={app}, "
                f"namespace={namespace}, endpoint={endpoint}, "
                f"method={method}. Error: {e}"
            )
            result.append(0.0)

    return float(np.mean(result)) if result else 0.0
