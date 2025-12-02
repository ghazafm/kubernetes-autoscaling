import ast
import json
from logging import Logger
from typing import Iterable, List, Tuple, Union


def parse_cpu_value(cpu_str: str, logger: Logger) -> float:
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
        logger.warning(f"Could not parse CPU value '{cpu_str}': {e}")
        return 0.0


def parse_memory_value(memory_str: str, logger: Logger) -> float:
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
        logger.warning(f"Could not parse memory value '{memory_str}': {e}")
        return 0.0


def normalize_endpoints(
    endpoints: Union[str, Iterable, None],
    default: Iterable[Tuple[str, str]] = (("/", "GET"), ("/docs", "GET")),
) -> List[Tuple[str, str]]:
    """
    Terima berbagai bentuk:
      - JSON string: [["/a","GET"],["/b","POST"]]
      - Python literal string: [("/a","GET"), ("/b","POST")]
      - List[tuple]: [("/a","GET"), ...]
      - List[str]: ["/a", "/b"]   -> method default "GET"
      - String tunggal: "/a"      -> method default "GET"
    Return: List[Tuple[str, str]]
    """
    if endpoints is None:
        endpoints = default

    # sudah iterable tuple/list?
    if isinstance(endpoints, (list, tuple)):
        out: List[Tuple[str, str]] = []
        for item in endpoints:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                ep = str(item[0])
                method = str(item[1]) if len(item) > 1 else "GET"
                out.append((ep, method))
            elif isinstance(item, str):
                out.append((item, "GET"))
        return out

    # string -> coba JSON dulu
    if isinstance(endpoints, str):
        s = endpoints.strip()
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(s)
                return normalize_endpoints(parsed, default)
            except Exception:
                print("Failed to parse endpoints JSON:", s)  # noqa: T201
        # fallback: anggap string tunggal path
        return [(s, "GET")]

    # fallback default
    return normalize_endpoints(default, default)
