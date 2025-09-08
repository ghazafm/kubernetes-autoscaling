import logging


def parse_cpu_value(cpu_str):
    """Parse CPU value from kubernetes format to cores (float)"""
    try:
        if cpu_str.endswith("m"):
            return float(cpu_str[:-1]) / 1000
        if cpu_str.endswith("n"):
            return float(cpu_str[:-1]) / 1000000000
        if cpu_str.endswith("u"):
            return float(cpu_str[:-1]) / 1000000
        return float(cpu_str)
    except (ValueError, IndexError) as e:
        logging.warning(f"Could not parse CPU value '{cpu_str}': {e}")
        return 0.0


def parse_memory_value(memory_str):
    """Parse memory value from kubernetes format to MB (float)"""
    try:
        if memory_str.endswith("Ki"):
            return float(memory_str[:-2]) / 1024
        if memory_str.endswith("Mi"):
            return float(memory_str[:-2])
        if memory_str.endswith("Gi"):
            return float(memory_str[:-2]) * 1024
        if memory_str.endswith("Ti"):
            return float(memory_str[:-2]) * 1024 * 1024
        return float(memory_str) / (1024 * 1024)
    except (ValueError, IndexError) as e:
        logging.warning(f"Could not parse memory value '{memory_str}': {e}")
        return 0.0
