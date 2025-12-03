import logging
import time

import numpy as np
from prometheus_api_client import PrometheusConnect


def wait_for_pods_ready(
    prometheus: PrometheusConnect,
    namespace: str,
    deployment_name: str,
    timeout: int,
    wait_time: int,
    logger: logging.Logger,
):
    start_time = time.time()

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

    # Initialize with defaults to avoid UnboundLocalError
    desired = 0.0
    ready = 0.0

    while time.time() - start_time < timeout:
        desired_result = prometheus.custom_query(q_desired)
        ready_result = prometheus.custom_query(query=q_ready)

        # Prometheus scalar results are [timestamp, 'value_string']
        # Convert to float for proper numeric comparison
        try:
            desired = float(desired_result[1]) if desired_result else 0.0
            ready = float(ready_result[1]) if ready_result else 0.0
        except (IndexError, ValueError, TypeError) as e:
            logger.warning(f"Error parsing Prometheus result: {e}")
            time.sleep(1)
            continue

        if np.isnan(desired):
            logger.debug("Desired replicas returned NaN, waiting for metrics...")
            time.sleep(1)
            continue
        if np.isnan(ready):
            ready = 0.0

        if ready == desired and desired > 0:
            time.sleep(wait_time)
            return True, int(desired), int(ready)
        logger.debug(f"Waiting for pods to be ready: {ready}/{desired}")
        time.sleep(1)
    time.sleep(wait_time)
    # Final NaN check before returning
    if np.isnan(desired):
        desired = 0.0
    if np.isnan(ready):
        ready = 0.0
    return False, int(desired), int(ready)
