def calculate_distance(
    cpu: float,
    memory: float,
    max_cpu: float,
    min_cpu: float,
    max_memory: float,
    min_memory: float,
) -> tuple[float, float]:
    cpu_bandwidth = max_cpu - min_cpu

    if cpu < min_cpu:
        cpu_distance = (cpu - min_cpu) / cpu_bandwidth
    elif cpu > max_cpu:
        cpu_distance = (cpu - max_cpu) / cpu_bandwidth
    else:
        cpu_distance = 0.0

    cpu_relative = (cpu - min_cpu) / cpu_bandwidth

    memory_bandwidth = max_memory - min_memory

    if memory < min_memory:
        memory_distance = (memory - min_memory) / memory_bandwidth
    elif memory > max_memory:
        memory_distance = (memory - max_memory) / memory_bandwidth
    else:
        memory_distance = 0.0

    memory_relative = (memory - min_memory) / memory_bandwidth

    return cpu_relative, memory_relative, cpu_distance, memory_distance
