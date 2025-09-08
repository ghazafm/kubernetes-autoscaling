import logging
import time

import numpy as np
from kubernetes.client.api import CoreV1Api, CustomObjectsApi

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
    api: CustomObjectsApi,
    core: CoreV1Api,
) -> tuple[float, float, int]:
    """
    Returns (cpu_usage_mean, memory_usage_mean, replica_count)
    cpu in %, memory in %, averaged over matched pods.
    """
    counter = 0
    cpu_usage = []
    memory_usage = []

    while counter < timeout:
        counter += 1
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

        collected = 0
        for item in target_metric_items:
            pod_name = item["metadata"]["name"]
            pod_obj = pod_spec_by_name.get(pod_name)
            if not pod_obj:
                logging.debug(f"Pod spec not found for {pod_name}, skipping")
                continue

            cpu_pct, mem_pct = calculate_usage(item, pod_obj)
            cpu_usage.append(cpu_pct)
            memory_usage.append(mem_pct)
            collected += 1
            if collected >= replicas:
                break

        if collected >= replicas:
            break

        time.sleep(1)

    cpu_vals = np.array(cpu_usage, dtype=float)
    mem_vals = np.array(memory_usage, dtype=float)
    cpu_mean = float(np.nanmean(cpu_vals)) if np.any(~np.isnan(cpu_vals)) else 0.0
    mem_mean = float(np.nanmean(mem_vals)) if np.any(~np.isnan(mem_vals)) else 0.0

    return cpu_mean, mem_mean, len(cpu_usage)


def get_response_time():
    return np.random.randint(50, 300)
