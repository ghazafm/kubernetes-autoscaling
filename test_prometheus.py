#!/usr/bin/env python3
"""
Prometheus Query Test Script for RL Autoscaler
Tests all queries used by the agent in real-time
"""

from datetime import datetime

from prometheus_api_client import PrometheusConnect

# Configuration
PROMETHEUS_URL = "http://10.34.4.150:30080/monitoring"
NAMESPACE = "default"
DEPLOYMENT = "flask-app"


def test_prometheus_queries():
    print("🔍 Testing Prometheus Queries for RL Autoscaler")
    print("=" * 60)
    print(f"Prometheus: {PROMETHEUS_URL}")
    print(f"Namespace: {NAMESPACE}")
    print(f"Deployment: {DEPLOYMENT}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # Connect to Prometheus
    prom = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)

    try:
        # Test connection
        print("🔗 Test 0: Prometheus Connection")
        is_connected = prom.check_prometheus_connection()
        print(f"   Result: {'✅ Connected' if is_connected else '❌ Failed'}")
        print("")

        if not is_connected:
            print("❌ Cannot connect to Prometheus. Check URL and network.")
            return

        # Test 1: Pod Count
        print("📊 Test 1: Current Pod Count")
        query = (
            f'count(kube_pod_info{{namespace="{NAMESPACE}",pod=~"{DEPLOYMENT}-.*"}})'
        )
        result = prom.custom_query(query)
        if result:
            pod_count = int(float(result[0]["value"][1]))
            print(f"   Result: {pod_count} pods")
        else:
            print("   Result: No data")
        print("")

        # Test 2: CPU Usage per Pod
        print("🔥 Test 2: CPU Usage per Pod (last 1m)")
        query = f"""
        sum by (pod) (
          rate(container_cpu_usage_seconds_total{{
            namespace="{NAMESPACE}",
            pod=~"{DEPLOYMENT}-.*",
            container!="",
            container!="POD"
          }}[1m])
        )
        """
        result = prom.custom_query(query)
        if result:
            print(f"   Found {len(result)} pods with CPU data:")
            for i, pod_data in enumerate(result[:5]):  # Show first 5
                pod_name = pod_data["metric"]["pod"]
                cpu_cores = float(pod_data["value"][1])
                print(
                    f"     {i + 1}. {pod_name}: {cpu_cores:.4f} cores ({cpu_cores * 1000:.0f}m)"
                )
            if len(result) > 5:
                print(f"     ... and {len(result) - 5} more pods")
        else:
            print("   Result: No data")
        print("")

        # Test 3: Memory Usage per Pod
        print("💾 Test 3: Memory Usage per Pod")
        query = f"""
        sum by (pod) (
          container_memory_working_set_bytes{{
            namespace="{NAMESPACE}",
            pod=~"{DEPLOYMENT}-.*",
            container!="",
            container!="POD"
          }}
        )
        """
        result = prom.custom_query(query)
        if result:
            print(f"   Found {len(result)} pods with Memory data:")
            for i, pod_data in enumerate(result[:5]):  # Show first 5
                pod_name = pod_data["metric"]["pod"]
                mem_bytes = float(pod_data["value"][1])
                mem_mi = mem_bytes / 1024 / 1024
                print(f"     {i + 1}. {pod_name}: {mem_mi:.2f} MiB")
            if len(result) > 5:
                print(f"     ... and {len(result) - 5} more pods")
        else:
            print("   Result: No data")
        print("")

        # Test 4: CPU Limits
        print("📏 Test 4: CPU Limits per Pod")
        query = f"""
        sum by (pod) (
          kube_pod_container_resource_limits{{
            namespace="{NAMESPACE}",
            pod=~"{DEPLOYMENT}-.*",
            resource="cpu",
            unit="core"
          }}
        )
        """
        result = prom.custom_query(query)
        if result:
            cpu_limit = float(result[0]["value"][1])
            print(f"   Result: {cpu_limit} cores ({cpu_limit * 1000:.0f}m) per pod")
            print(f"   Found limits for {len(result)} pods")
        else:
            print("   Result: No data")
        print("")

        # Test 5: Memory Limits
        print("📏 Test 5: Memory Limits per Pod")
        query = f"""
        sum by (pod) (
          kube_pod_container_resource_limits{{
            namespace="{NAMESPACE}",
            pod=~"{DEPLOYMENT}-.*",
            resource="memory",
            unit="byte"
          }}
        )
        """
        result = prom.custom_query(query)
        if result:
            mem_limit_bytes = float(result[0]["value"][1])
            mem_limit_mi = mem_limit_bytes / 1024 / 1024
            print(f"   Result: {mem_limit_mi:.0f} MiB per pod")
            print(f"   Found limits for {len(result)} pods")
        else:
            print("   Result: No data")
        print("")

        # Test 6: Request Rate
        print("🚀 Test 6: Request Rate (req/s)")
        query = f"""
        sum(
          rate(http_requests_total{{
            namespace="{NAMESPACE}",
            pod=~"{DEPLOYMENT}-.*"
          }}[1m])
        )
        """
        result = prom.custom_query(query)
        if result:
            rps = float(result[0]["value"][1])
            print(f"   Result: {rps:.2f} req/s")
        else:
            print("   Result: No data (metrics not available yet)")
        print("")

        # Test 7: Response Time (p95)
        print("⏱️  Test 7: Response Time p95 (last 1m)")
        query = f"""
        histogram_quantile(0.95,
          sum by (le) (
            rate(http_request_duration_seconds_bucket{{
              namespace="{NAMESPACE}",
              pod=~"{DEPLOYMENT}-.*"
            }}[1m])
          )
        ) * 1000
        """
        result = prom.custom_query(query)
        if result:
            p95_ms = float(result[0]["value"][1])
            print(f"   Result: {p95_ms:.2f} ms")
        else:
            print("   Result: No data (metrics not available yet)")
        print("")

        # Test 8: Average CPU Usage %
        print("📈 Test 8: Average CPU Usage %")
        query = f"""
        avg(
          sum by (pod) (
            rate(container_cpu_usage_seconds_total{{
              namespace="{NAMESPACE}",
              pod=~"{DEPLOYMENT}-.*",
              container!="",
              container!="POD"
            }}[1m])
          ) / on(pod) group_left()
          sum by (pod) (
            kube_pod_container_resource_limits{{
              namespace="{NAMESPACE}",
              pod=~"{DEPLOYMENT}-.*",
              resource="cpu"
            }}
          ) * 100
        )
        """
        result = prom.custom_query(query)
        if result:
            cpu_pct = float(result[0]["value"][1])
            print(f"   Result: {cpu_pct:.2f}%")
        else:
            print("   Result: No data")
        print("")

        # Test 9: Average Memory Usage %
        print("📈 Test 9: Average Memory Usage %")
        query = f"""
        avg(
          sum by (pod) (
            container_memory_working_set_bytes{{
              namespace="{NAMESPACE}",
              pod=~"{DEPLOYMENT}-.*",
              container!="",
              container!="POD"
            }}
          ) / on(pod) group_left()
          sum by (pod) (
            kube_pod_container_resource_limits{{
              namespace="{NAMESPACE}",
              pod=~"{DEPLOYMENT}-.*",
              resource="memory"
            }}
          ) * 100
        )
        """
        result = prom.custom_query(query)
        if result:
            mem_pct = float(result[0]["value"][1])
            print(f"   Result: {mem_pct:.2f}%")
        else:
            print("   Result: No data")
        print("")

        # Test 10: Ready Pods (used by RL agent)
        print("✅ Test 10: Ready Pods Count")
        query = f"""
        count(
          (kube_pod_status_ready{{namespace="{NAMESPACE}", condition="true"}} == 1)
          and on(pod)
          (
            label_replace(
              kube_pod_owner{{namespace="{NAMESPACE}", owner_kind="ReplicaSet"}},
              "replicaset", "$1", "owner_name", "(.*)"
            )
            * on(namespace, replicaset) group_left(owner_name)
            kube_replicaset_owner{{
              namespace="{NAMESPACE}",
              owner_kind="Deployment",
              owner_name="{DEPLOYMENT}"
            }}
          )
        )
        """
        result = prom.custom_query(query)
        if result:
            ready_pods = int(float(result[0]["value"][1]))
            print(f"   Result: {ready_pods} ready pods")
        else:
            print("   Result: No data")
        print("")

        print("=" * 60)
        print("✅ All Prometheus query tests completed!")
        print("")
        print("💡 Next steps:")
        print("   1. View queries in browser: " + PROMETHEUS_URL + "/graph")
        print("   2. Check full documentation: PROMETHEUS_QUERIES.md")
        print("   3. Run RL agent: cd agent && python run.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_prometheus_queries()
