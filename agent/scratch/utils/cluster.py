import time
from logging import Logger

from prometheus_api_client import PrometheusConnect

READY_RESULT_MIN_LENGTH = 1
RESULT_MIN_LENGTH = 2


def _extract_scalar_value(result, logger: Logger, label: str) -> int | None:  # noqa: PLR0912
    raw = None

    # Case 1: list
    if isinstance(result, list):
        if not result:
            logger.debug(f"{label}: empty list result from Prometheus")
            return None

        first = result[0]

        # 1a) standard: [{"metric":..., "value": ["ts", "3"]}]
        if isinstance(first, dict) and "value" in first:
            value = first["value"]
            if isinstance(value, (list, tuple)) and len(value) >= RESULT_MIN_LENGTH:
                raw = value[1]
            else:
                raw = value

        elif len(result) >= RESULT_MIN_LENGTH and not isinstance(first, dict):
            raw = result[1]

        else:
            logger.debug(f"{label}: unrecognized list format: {result!r}")
            return None

    # Case 2: scalar langsung
    elif isinstance(result, (int, float, str)):
        raw = result

    # Case 3: dict tapi tidak dalam list
    elif isinstance(result, dict):
        value = result.get("value")
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            raw = value[1]
        else:
            raw = value

    else:
        logger.debug(f"{label}: unsupported result type: {type(result)} -> {result!r}")
        return None

    if raw in {None, "NaN", ""}:
        logger.debug(f"{label}: raw value invalid: {raw!r}")
        return None

    try:
        return int(float(str(raw)))
    except Exception as e:
        logger.error(f"{label}: failed to parse {raw!r} -> int: {e}")
        return None


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
            desired_result = prometheus.custom_query(query=q_desired)
            logger.debug(
                "wait_for_pods_ready: raw desired_result=%r (type=%s)",
                desired_result,
                type(desired_result),
            )

            desired_replicas_prom = _extract_scalar_value(
                desired_result, logger, "desired_replicas"
            )
            if desired_replicas_prom is None:
                time.sleep(1)
                continue

            if desired_replicas_prom != desired_replicas:
                logger.debug(
                    "wait_for_pods_ready: desired_replicas mismatch, "
                    f"expected {desired_replicas}, got {desired_replicas_prom}"
                )
                time.sleep(1)
                continue

            logger.debug(
                "wait_for_pods_ready: desired_replicas matched: "
                f"{desired_replicas_prom}"
            )

            ready_result = prometheus.custom_query(query=q_ready)
            logger.debug(
                "wait_for_pods_ready: raw ready_result=%r (type=%s)",
                ready_result,
                type(ready_result),
            )

            ready_replicas_value = _extract_scalar_value(
                ready_result, logger, "ready_replicas"
            )
            ready_replicas = ready_replicas_value or 0

            logger.debug(f"wait_for_pods_ready: ready_replicas={ready_replicas}")
            if ready_replicas == desired_replicas > 0:
                logger.debug("wait_for_pods_ready: pods are ready")
                return True, desired_replicas, ready_replicas

            logger.debug(
                "wait_for_pods_ready: not ready yet, "
                f"{ready_replicas}/{desired_replicas} ready"
            )
            time.sleep(1)

        except Exception as e:
            logger.error(f"Error checking pod readiness: {e}")
            time.sleep(1)

    logger.warning(f"Timeout waiting for pods to be ready after {timeout}s")
    return False, desired_replicas, ready_replicas
