import time
from logging import Logger

from prometheus_api_client import PrometheusConnect


def wait_for_pods_ready(
    prometheus: PrometheusConnect,
    deployment_name: str,
    desired_replicas: int,
    namespace: str,
    timeout: int,
    logger: Logger,
) -> tuple[bool, int, int]:
    """Wait for pods to be ready after scaling operation."""
    start_time = time.time()
    ready_replicas = 0

    scope_ready = f"""
        (kube_pod_status_ready{{namespace="{namespace}", condition="true"}} == 1)
        and on(pod)
        (
          label_replace(
            kube_pod_owner{{namespace="{namespace}", owner_kind="ReplicaSet"}},
            "replicaset", "$1", "owner_name", "(.*)"
          )
          * on(namespace, replicaset) group_left(owner_name)
            kube_replicaset_owner{{
              namespace="{namespace}", owner_kind="Deployment", owner_name="{deployment_name}"
            }}
        )
    """  # noqa: E501
    q_desired = f"""
    scalar(
      sum(
        kube_deployment_spec_replicas{{namespace="{namespace}",
        deployment="{deployment_name}"}}
        )
    )
    """

    q_ready = f"""
      scalar(sum({scope_ready}))
    """

    logger.debug(f"wait_for_pods_ready: q_ready={q_ready}")
    logger.debug(f"wait_for_pods_ready: q_desired={q_desired}")

    while time.time() - start_time < timeout:
        try:
            desired_replicas_prom = int(prometheus.custom_query(query=q_desired)[1])
            if desired_replicas_prom != desired_replicas:
                logger.debug(
                    f"wait_for_pods_ready: desired_replicas mismatch, "
                    f"expected {desired_replicas}, got {desired_replicas_prom}"
                )
                time.sleep(1)
                continue

            logger.debug(
                "wait_for_pods_ready: desired_replicas "
                f"matched: {desired_replicas_prom}"
            )

            ready_replicas = int(prometheus.custom_query(query=q_ready)[1])
            logger.debug(f"wait_for_pods_ready: ready_replicas={ready_replicas}")
            if ready_replicas == desired_replicas > 0:
                logger.debug("wait_for_pods_ready: pods are ready")
                return True, desired_replicas, ready_replicas
            logger.debug(
                f"wait_for_pods_ready: not ready yet, "
                f"{ready_replicas}/{desired_replicas} ready"
            )

            time.sleep(1)

        except Exception as e:
            logger.error(f"Error checking pod readiness: {e}")
            time.sleep(1)

    logger.warning(f"Timeout waiting for pods to be ready after {timeout}s")

    return False, desired_replicas, ready_replicas
