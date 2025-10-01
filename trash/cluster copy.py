import time
from logging import Logger

from prometheus_api_client import PrometheusApiClientException, PrometheusConnect


def wait_for_pods_ready(
    prometheus: PrometheusConnect,
    deployment_name: str,
    interval: int,
    namespace: str,
    timeout: int,
    logger: Logger,
) -> tuple[bool, int, int]:
    """Wait until all desired pods are Ready AND their metrics are present in Prometheus

    Uses kube-state-metrics via Prometheus:
      - desired replicas:  kube_deployment_spec_replicas
      - ready pods:        kube_pod_status_ready joined through ReplicaSet->Deployment

    Also verifies per-pod metric presence (for the *same* pods):
      - CPU usage:         rate(container_cpu_usage_seconds_total[win])
      - Memory usage:      container_memory_working_set_bytes
      - Denominator:       CPU limit OR request, Memory limit OR request

    Returns (ready, desired_replicas, ready_replicas).
    """
    prom: PrometheusConnect = prometheus
    if not hasattr(prom, "custom_query"):
        logger.error("wait_for_pods_ready: 'prometheus' must be a PrometheusConnect")
    start_time = time.time()
    sleep_s = 3
    desired_replicas = 0
    ready_replicas = 0

    # Define scope for ready pods belonging to this deployment
    scope_ready = f"""
      (
        (kube_pod_status_ready{{namespace="{namespace}", condition="true"}} == 1)
        and on(pod)
        (
          kube_pod_owner{{namespace="{namespace}", owner_kind="ReplicaSet"}}
          * on(owner_name) group_left()
          kube_replicaset_owner{{
            namespace="{namespace}", owner_kind="Deployment",
            owner_name="{deployment_name}"
          }}
        )
      )
    """

    # PromQL for desired replicas (Deployment spec)
    q_desired = f"""
      kube_deployment_spec_replicas{{namespace="{namespace}", deployment="{deployment_name}"}}
    """  # noqa: E501

    # PromQL for Ready pods that belong to this Deployment (via owner chain)
    q_ready = f"""
      sum({scope_ready})
    """

    # Helper to run a scalar PromQL safely
    def _query_scalar(q: str) -> int:
        try:
            res = prom.custom_query(q) or []
            return int(float(res[0]["value"][1])) if res else 0
        except PrometheusApiClientException as e:
            logger.warning(f"Prometheus query failed: {e}")
            return 0
        except Exception as e:
            logger.warning(f"Unexpected error during Prom readiness check: {e}")
            return 0

    pod_re = f"^{deployment_name}-.*"

    q_metrics_ok = f"""
      count(
        (
          sum by (pod)(
            rate(container_cpu_usage_seconds_total{{
              namespace="{namespace}", pod=~"{pod_re}", container!="", container!="POD"
            }}[{interval}s])
          ) > 0
        )
        AND on(pod)
        (
          sum by (pod)(
            container_memory_working_set_bytes{{
              namespace="{namespace}", pod=~"{pod_re}", container!="", container!="POD"
            }}
          ) > 0
        )
        AND on(pod)
        (
          sum by (pod)(
            kube_pod_container_resource_limits{{
              namespace="{namespace}", pod=~"{pod_re}", resource="cpu", unit="core"
            }}
          ) > 0
          OR
          sum by (pod)(
            kube_pod_container_resource_requests{{
              namespace="{namespace}", pod=~"{pod_re}", resource="cpu", unit="core"
            }}
          ) > 0
        )
        AND on(pod)
        (
          sum by (pod)(
            kube_pod_container_resource_limits{{
              namespace="{namespace}", pod=~"{pod_re}", resource="memory", unit="byte"
            }}
          ) > 0
          OR
          sum by (pod)(
            kube_pod_container_resource_requests{{
              namespace="{namespace}", pod=~"{pod_re}", resource="memory", unit="byte"
            }}
          ) > 0
        )
        AND on(pod)
        {scope_ready}
      )
    """

    while time.time() - start_time < timeout:
        desired_replicas = _query_scalar(q_desired)
        ready_replicas = _query_scalar(q_ready)

        logger.debug(
            f"[wait_for_pods_ready/prom] desired={desired_replicas}, "
            f"ready={ready_replicas}"
        )

        # Only when Kubernetes (via Prom) says Ready == Desired...
        if desired_replicas > 0 and ready_replicas == desired_replicas:
            # ...verify Prometheus has metrics for ALL those pods
            try:
                res = prom.custom_query(q_metrics_ok) or []
                metrics_ok = int(float(res[0]["value"][1])) if res else 0
            except Exception as e:
                logger.debug(f"[wait_for_pods_ready] metrics-ready query failed: {e}")
                metrics_ok = 0

            logger.debug(
                f"[wait_for_pods_ready/prom] metrics_ok={metrics_ok} "
                f"(need {desired_replicas})"
            )

            if metrics_ok >= desired_replicas:
                return True, desired_replicas, ready_replicas

        time.sleep(sleep_s)

    logger.warning(f"Timeout waiting for pods/metrics to be ready after {timeout}s")
    return False, desired_replicas, ready_replicas


def wait_for_pods_ready(
    cluster: PrometheusConnect,
    deployment_name: str,
    namespace: str,
    timeout: int,
    logger: Logger,
) -> tuple[bool, int, int]:
    """Wait for pods to be ready after scaling operation."""
    start_time = time.time()
    desired_replicas = 0
    ready_replicas = 0
    while time.time() - start_time < timeout:
        try:
            deployment = cluster.read_namespaced_deployment(
                name=deployment_name, namespace=namespace
            )
            status = getattr(deployment, "status", None)
            spec = getattr(deployment, "spec", None)

            if status is not None:
                ready_replicas = getattr(status, "ready_replicas", 0) or 0
            else:
                ready_replicas = 0

            if spec is not None:
                desired_replicas = getattr(spec, "replicas", 0) or 0
            else:
                desired_replicas = 0

            if ready_replicas == desired_replicas > 0:
                return True, desired_replicas, ready_replicas

            time.sleep(5)

        except Exception as e:
            logger.error(f"Error checking pod readiness: {e}")
            time.sleep(5)

    logger.warning(f"Timeout waiting for pods to be ready after {timeout}s")

    return False, desired_replicas, ready_replicas
