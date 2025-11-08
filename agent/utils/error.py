"""
Enhanced metrics collection with comprehensive error tracking.

This module extends the base metrics.py with better error detection including:
- HTTP 5xx errors from application
- Connection failures (refused, reset, timeout)
- Pod readiness issues
- Service unavailability
"""

import logging

from prometheus_api_client import PrometheusApiClientException, PrometheusConnect


def _get_error_rate(
    prometheus: PrometheusConnect,
    deployment_name: str,
    namespace: str = "default",
    interval: int = 15,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
) -> tuple[float, dict]:
    """
    Get comprehensive error rate including connection-level failures.

    Returns:
        tuple: (total_error_rate_percentage, error_breakdown_dict)

    Error breakdown includes:
        - application_5xx: HTTP 5xx errors from app
        - connection_failures: Connection refused, resets, timeouts
        - pod_not_ready: Requests to non-ready pods
        - total_requests: Total request attempts
    """
    error_breakdown = {
        "application_5xx": 0.0,
        "connection_failures": 0.0,
        "pod_not_ready": 0.0,
        "total_requests": 0.0,
        "total_errors": 0.0,
        "error_rate_pct": 0.0,
    }

    try:
        # 1. Application-level 5xx errors
        app_error_query = f"""
            sum(
                rate(http_requests_total{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    http_status=~"5.."
                }}[{interval}s])
            )
        """

        # 2. Total successful requests (got HTTP response)
        total_app_requests_query = f"""
            sum(
                rate(http_requests_total{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*"
                }}[{interval}s])
            )
        """

        # 3. Connection failures (if using service mesh or nginx ingress)
        # This query works with nginx-ingress-controller
        connection_failure_query = f"""
            sum(
                rate(nginx_ingress_controller_requests{{
                    exported_namespace="{namespace}",
                    service=~"{deployment_name}.*",
                    status=~"0|503|504"
                }}[{interval}s])
            )
        """

        # 4. Total attempts (if using ingress/service mesh)
        total_attempts_query = f"""
            sum(
                rate(nginx_ingress_controller_requests{{
                    exported_namespace="{namespace}",
                    service=~"{deployment_name}.*"
                }}[{interval}s])
            )
        """

        # Query metrics
        app_errors_result = prometheus.custom_query(app_error_query)
        total_app_result = prometheus.custom_query(total_app_requests_query)

        app_errors = (
            float(app_errors_result[0]["value"][1])
            if app_errors_result and len(app_errors_result) > 0
            else 0.0
        )

        total_app_requests = (
            float(total_app_result[0]["value"][1])
            if total_app_result and len(total_app_result) > 0
            else 0.0
        )

        error_breakdown["application_5xx"] = app_errors
        error_breakdown["total_requests"] = total_app_requests

        # Try to get connection-level failures (may not be available)
        try:
            conn_failure_result = prometheus.custom_query(connection_failure_query)
            total_attempts_result = prometheus.custom_query(total_attempts_query)

            conn_failures = (
                float(conn_failure_result[0]["value"][1])
                if conn_failure_result and len(conn_failure_result) > 0
                else 0.0
            )

            total_attempts = (
                float(total_attempts_result[0]["value"][1])
                if total_attempts_result and len(total_attempts_result) > 0
                else total_app_requests  # Fallback to app requests
            )

            error_breakdown["connection_failures"] = conn_failures
            error_breakdown["total_requests"] = max(total_attempts, total_app_requests)

        except (PrometheusApiClientException, Exception) as e:
            logger.debug(
                f"Connection-level metrics not available (nginx-ingress may not be "
                f"installed): {e}"
            )
            # Not an error - just means we only have app-level metrics

        # Calculate total error rate
        total_errors = (
            error_breakdown["application_5xx"] + error_breakdown["connection_failures"]
        )
        total_requests = error_breakdown["total_requests"]

        error_breakdown["total_errors"] = total_errors

        if total_requests > 0:
            error_rate_pct = (total_errors / total_requests) * 100.0
            # Handle edge case where errors might be slightly > requests due to timing
            error_rate_pct = min(error_rate_pct, 100.0)
        else:
            # No requests = no errors
            error_rate_pct = 0.0

        error_breakdown["error_rate_pct"] = error_rate_pct

        logger.debug(
            f"Comprehensive error rate: {error_rate_pct:.2f}% "
            f"(App errors: {app_errors:.2f}/s, "
            f"Conn failures: {error_breakdown['connection_failures']:.2f}/s, "
            f"Total requests: {total_requests:.2f}/s)"
        )

        return error_rate_pct, error_breakdown

    except Exception as e:
        logger.error(f"Failed to calculate comprehensive error rate: {e}")
        # Return conservative estimate
        return 0.0, error_breakdown


def _get_pod_readiness_error_estimate(
    prometheus: PrometheusConnect,
    deployment_name: str,
    namespace: str = "default",
    desired_replicas: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
) -> float:
    """
    Estimate error rate contribution from non-ready pods.

    When pods are not ready, incoming requests may fail at the connection level.
    This estimates the percentage of requests that might fail due to pod readiness.

    Returns:
        float: Estimated additional error rate percentage (0-100)
    """
    try:
        # Query for number of ready pods
        # kube_pod_status_ready has condition label with values: true/false/unknown
        # When condition="true" and value=1, the pod is ready
        ready_pods_query = f"""
            sum(
                kube_pod_status_ready{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    condition="true"
                }}
            )
        """

        # Query for total pods - count unique pods
        # Use kube_pod_status_ready with any condition to get unique pod count
        total_pods_query = f"""
            count(
                kube_pod_status_ready{{
                    namespace="{namespace}",
                    pod=~"{deployment_name}-.*",
                    condition="true"
                }}
            )
        """

        ready_result = prometheus.custom_query(ready_pods_query)
        total_result = prometheus.custom_query(total_pods_query)

        ready_pods = (
            int(float(ready_result[0]["value"][1]))
            if ready_result and len(ready_result) > 0
            else desired_replicas
        )

        total_pods = (
            int(float(total_result[0]["value"][1]))
            if total_result and len(total_result) > 0
            else desired_replicas
        )

        if total_pods == 0:
            logger.warning("No pods found for readiness check")
            return 0.0

        # Calculate readiness ratio
        readiness_ratio = ready_pods / total_pods

        # Estimate error contribution
        # If 50% of pods are not ready, assume ~25% of requests might fail
        # (not all requests will hit non-ready pods due to service load balancing)
        not_ready_ratio = 1.0 - readiness_ratio
        estimated_error_contribution = not_ready_ratio * 50.0  # Conservative estimate

        NOT_READY_RATIO_THRESHOLD = 0.1
        if not_ready_ratio > NOT_READY_RATIO_THRESHOLD:  # More than 10% not ready
            logger.warning(
                f"Pod readiness issue: {ready_pods}/{total_pods} ready "
                f"(estimated error contribution: {estimated_error_contribution:.1f}%)"
            )

        return estimated_error_contribution

    except Exception as e:
        logger.debug(f"Could not estimate pod readiness errors: {e}")
        return 0.0


def get_error_rate(
    prometheus: PrometheusConnect,
    deployment_name: str,
    namespace: str = "default",
    interval: int = 15,
    desired_replicas: int = 1,
    logger: logging.Logger = logging.getLogger(__name__),  # noqa: B008
) -> tuple[float, dict]:
    """
    Get enhanced error rate with multiple detection methods.

    Combines:
    1. Application 5xx errors (from http_requests_total)
    2. Connection failures (from nginx-ingress if available)
    3. Pod readiness issues (from kube_pod_status_ready)

    Returns:
        tuple: (total_error_rate_percentage, detailed_breakdown)
    """
    # Get error rate from metrics
    error_rate, breakdown = _get_error_rate(
        prometheus=prometheus,
        deployment_name=deployment_name,
        namespace=namespace,
        interval=interval,
        logger=logger,
    )

    # Add pod readiness estimate
    readiness_error_estimate = _get_pod_readiness_error_estimate(
        prometheus=prometheus,
        deployment_name=deployment_name,
        namespace=namespace,
        desired_replicas=desired_replicas,
        logger=logger,
    )

    breakdown["pod_not_ready_estimate"] = readiness_error_estimate

    # Combine error rates (but don't double-count)
    # If we have actual connection failure metrics, use those
    # Otherwise, use readiness estimate as proxy
    if breakdown["connection_failures"] > 0:
        total_error_rate = error_rate
    else:
        # Add readiness estimate but cap at reasonable maximum
        total_error_rate = min(error_rate + readiness_error_estimate, 100.0)

    breakdown["final_error_rate_pct"] = total_error_rate

    logger.debug(
        f"Enhanced error rate: {total_error_rate:.2f}% "
        f"(base: {error_rate:.2f}%, readiness adj: {readiness_error_estimate:.1f}%)"
    )

    return total_error_rate, breakdown
