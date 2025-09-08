import logging
import time

from kubernetes.client.api import CoreV1Api


def wait_for_pods_ready(
    cluster: CoreV1Api, deployment_name: str, namespace: str, timeout: int
):
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
            logging.error(f"Error checking pod readiness: {e}")
            time.sleep(5)

    logging.warning(f"Timeout waiting for pods to be ready after {timeout}s")

    return False, desired_replicas, ready_replicas
