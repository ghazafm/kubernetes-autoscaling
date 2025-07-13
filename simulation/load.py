"""
Simple Adaptive Load Tester

This script creates variable load patterns for Kubernetes autoscaling RL training.
It runs independently and can be controlled via command-line arguments.
"""

import argparse
import asyncio
import json
import logging
import math
import random  # nosec B404
import signal
import sys
import time
from datetime import datetime
from typing import Optional

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            f"load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
logger = logging.getLogger(__name__)


class SimpleLoadTester:
    """Simple load tester with adaptive patterns"""

    MAX_RESPONSE_TIMES = 1000

    def __init__(
        self,
        target_url: str,
        max_rps: int = 50,
        max_concurrent: int = 20,
        stress_probability: float = 0.1,
    ):
        self.target_url = target_url
        self.max_rps = max_rps
        self.max_concurrent = max_concurrent
        self.stress_probability = stress_probability
        self.session: Optional[aiohttp.ClientSession] = None
        self.running = False
        self.current_rps = 1.0
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "stress_requests": 0,
            "normal_requests": 0,
            "start_time": None,
            "response_times": [],
        }

    async def start(self):
        """Start the load tester"""
        if self.running:
            return

        logger.info(f"Starting load tester - Target: {self.target_url}")

        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=10)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        self.running = True
        self.stats["start_time"] = time.time()

    async def stop(self):
        """Stop the load tester"""
        if not self.running:
            return

        logger.info("Stopping load tester")
        self.running = False

        if self.session:
            await self.session.close()

    def get_request_url(self) -> str:
        """Get request URL (normal or stress endpoint)"""
        if random.random() < self.stress_probability:  # noqa: S311
            return (
                f"{self.target_url}/stress"
                if not self.target_url.endswith("/")
                else f"{self.target_url}stress"
            )
        return self.target_url

    async def make_request(self):
        """Make a single HTTP request"""
        if not self.session:
            return

        request_url = self.get_request_url()
        is_stress = "/stress" in request_url

        start_time = time.time()
        try:
            async with self.session.get(request_url) as response:
                await response.text()
                end_time = time.time()

                response_time = end_time - start_time
                self.stats["response_times"].append(response_time)
                if len(self.stats["response_times"]) > self.MAX_RESPONSE_TIMES:
                    self.stats["response_times"] = self.stats["response_times"][
                        -self.MAX_RESPONSE_TIMES :
                    ]
                if len(self.stats["response_times"]) > self.MAX_RESPONSE_TIMES:
                    self.stats["response_times"] = self.stats["response_times"][-1000:]

                if is_stress:
                    self.stats["stress_requests"] += 1
                else:
                    self.stats["normal_requests"] += 1

                OK = 200
                OK_RANGE = 300
                if OK <= response.status < OK_RANGE:
                    self.stats["successful_requests"] += 1
                else:
                    self.stats["failed_requests"] += 1

        except Exception as e:
            self.stats["total_requests"] += 1
            self.stats["failed_requests"] += 1
            logger.debug(f"Request failed: {e}")

    async def run_load_generation(self):
        """Main load generation loop"""
        active_tasks = set()

        while self.running:
            try:
                delay = 1.0 / self.current_rps if self.current_rps > 0 else 1.0

                active_tasks = {task for task in active_tasks if not task.done()}

                if len(active_tasks) < self.max_concurrent:
                    task = asyncio.create_task(self.make_request())
                    active_tasks.add(task)

                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in load generation: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    def get_stats(self):
        """Get current statistics"""
        elapsed = (
            time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        )
        success_rate = (
            self.stats["successful_requests"] / max(1, self.stats["total_requests"])
        ) * 100

        avg_response_time = 0
        if self.stats["response_times"]:
            avg_response_time = sum(self.stats["response_times"]) / len(
                self.stats["response_times"]
            )

        return {
            "elapsed_time": elapsed,
            "current_rps": self.current_rps,
            "total_requests": self.stats["total_requests"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "actual_rps": self.stats["total_requests"] / max(1, elapsed),
        }

    def print_stats(self):
        """Print current statistics"""
        stats = self.get_stats()
        logger.info(
            f"Stats - RPS: {stats['current_rps']:.1f} | "
            f"Total: {stats['total_requests']} | "
            f"Success: {stats['success_rate']:.1f}% | "
            f"Avg RT: {stats['avg_response_time']:.3f}s | "
            f"Actual RPS: {stats['actual_rps']:.1f}"
        )


class LoadPatternGenerator:
    """Generates different load patterns"""

    def __init__(self, base_rps: float = 5.0, max_rps: float = 50.0):
        self.base_rps = base_rps
        self.max_rps = max_rps
        self.start_time = time.time()

    def get_pattern_rps(self, pattern: str, elapsed_time: float) -> float:
        """Get RPS for a given pattern at elapsed time"""

        if pattern == "constant":
            return self.base_rps

        if pattern == "linear_ramp":
            progress = min(elapsed_time / 300, 1.0)
            return self.base_rps + (self.max_rps - self.base_rps) * progress

        if pattern == "sine_wave":
            wave = math.sin(elapsed_time * 2 * math.pi / 120)
            return self.base_rps + (self.max_rps - self.base_rps) * (wave + 1) / 2

        if pattern == "spike":
            cycle_time = elapsed_time % 180
            if 60 <= cycle_time <= 90:
                return self.max_rps
            return self.base_rps

        if pattern == "random":
            return self.base_rps + random.uniform(0.2, 1.0) * (  # noqa: S311
                self.max_rps - self.base_rps
            )

        if pattern == "burst":
            if random.random() < 0.1:  # noqa: S311
                return self.max_rps
            return self.base_rps * 0.3

        if pattern == "realistic":
            daily_progress = (elapsed_time / 600) * 24
            hour = daily_progress % 24

            if 9 <= hour <= 17:
                multiplier = 1.0
            elif 6 <= hour <= 9 or 17 <= hour <= 22:
                multiplier = 0.6
            else:
                multiplier = 0.2

            return self.base_rps + (self.max_rps - self.base_rps) * multiplier

        return self.base_rps

    def get_stress_probability(self, pattern: str, elapsed_time: float) -> float:
        """Get stress probability for a given pattern"""

        if pattern == "burst":
            return 0.25

        if pattern == "spike":
            cycle_time = elapsed_time % 180
            if 60 <= cycle_time <= 90:
                return 0.30
            return 0.05

        if pattern == "realistic":
            daily_progress = (elapsed_time / 600) * 24
            hour = daily_progress % 24

            if 9 <= hour <= 17:
                return 0.15
            if 6 <= hour <= 9 or 17 <= hour <= 22:
                return 0.10
            return 0.05

        return 0.10


async def run_adaptive_load_test(args):
    """Run adaptive load test with changing patterns"""

    load_tester = SimpleLoadTester(
        target_url=args.url, max_rps=args.max_rps, max_concurrent=args.max_concurrent
    )

    pattern_generator = LoadPatternGenerator(
        base_rps=args.base_rps, max_rps=args.max_rps
    )

    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        load_tester.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await load_tester.start()

        load_task = asyncio.create_task(load_tester.run_load_generation())

        patterns = [
            "constant",
            "linear_ramp",
            "sine_wave",
            "spike",
            "random",
            "burst",
            "realistic",
        ]
        pattern_index = 0
        pattern_start_time = time.time()
        pattern_duration = args.pattern_duration

        logger.info(f"Starting adaptive load test for {args.duration} seconds")

        test_start_time = time.time()
        last_stats_time = time.time()

        while load_tester.running and (time.time() - test_start_time) < args.duration:
            current_time = time.time()
            elapsed_time = current_time - test_start_time

            if (
                args.adaptive
                and (current_time - pattern_start_time) >= pattern_duration
            ):
                pattern_index = (pattern_index + 1) % len(patterns)
                pattern_start_time = current_time
                logger.info(f"Switching to pattern: {patterns[pattern_index]}")

            current_pattern = patterns[pattern_index] if args.adaptive else args.pattern

            pattern_elapsed = current_time - pattern_start_time
            target_rps = pattern_generator.get_pattern_rps(
                current_pattern, pattern_elapsed
            )

            if args.dynamic_stress:
                stress_prob = pattern_generator.get_stress_probability(
                    current_pattern, pattern_elapsed
                )
                load_tester.stress_probability = stress_prob

            load_tester.current_rps = target_rps

            if current_time - last_stats_time >= 30:
                load_tester.print_stats()
                last_stats_time = current_time

            await asyncio.sleep(5)

        load_tester.running = False
        await load_task

    except Exception as e:
        logger.error(f"Load test failed: {e}")

    finally:
        await load_tester.stop()

        final_stats = load_tester.get_stats()
        logger.info("Final Statistics:")
        logger.info(f"  Duration: {final_stats['elapsed_time']:.1f}s")
        logger.info(f"  Total Requests: {final_stats['total_requests']}")
        logger.info(f"  Successful: {final_stats['successful_requests']}")
        logger.info(f"  Failed: {final_stats['failed_requests']}")
        logger.info(f"  Normal Requests: {final_stats['normal_requests']}")
        logger.info(f"  Stress Requests: {final_stats['stress_requests']}")
        logger.info(f"  Success Rate: {final_stats['success_rate']:.1f}%")
        logger.info(f"  Stress Rate: {final_stats['stress_rate']:.1f}%")
        logger.info(f"  Average Response Time: {final_stats['avg_response_time']:.3f}s")
        logger.info(f"  Average RPS: {final_stats['actual_rps']:.1f}")

        stats_file = f"load_test_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, "w") as f:
            json.dump(final_stats, f, indent=2)
        logger.info(f"Stats saved to {stats_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple Adaptive Load Tester")

    parser.add_argument("--url", required=True, help="Target URL to test")
    parser.add_argument(
        "--duration",
        type=int,
        default=600,
        help="Test duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--base-rps",
        type=float,
        default=5.0,
        help="Base requests per second (default: 5.0)",
    )
    parser.add_argument(
        "--max-rps",
        type=float,
        default=50.0,
        help="Maximum requests per second (default: 50.0)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Maximum concurrent requests (default: 20)",
    )

    parser.add_argument(
        "--pattern",
        choices=[
            "constant",
            "linear_ramp",
            "sine_wave",
            "spike",
            "random",
            "burst",
            "realistic",
        ],
        default="sine_wave",
        help="Load pattern to use (default: sine_wave)",
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="Enable adaptive pattern switching"
    )
    parser.add_argument(
        "--pattern-duration",
        type=int,
        default=120,
        help="Duration of each pattern in adaptive mode (default: 120)",
    )
    parser.add_argument(
        "--stress-probability",
        type=float,
        default=0.1,
        help="Probability of using stress endpoint (default: 0.1)",
    )
    parser.add_argument(
        "--dynamic-stress",
        action="store_true",
        help="Enable dynamic stress probability based on pattern",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Load Test Configuration:")
    logger.info(f"  Target URL: {args.url}")
    logger.info(f"  Duration: {args.duration}s")
    logger.info(f"  Base RPS: {args.base_rps}")
    logger.info(f"  Max RPS: {args.max_rps}")
    logger.info(f"  Max Concurrent: {args.max_concurrent}")
    logger.info(f"  Pattern: {args.pattern}")
    logger.info(f"  Adaptive: {args.adaptive}")
    if args.adaptive:
        logger.info(f"  Pattern Duration: {args.pattern_duration}s")

    try:
        asyncio.run(run_adaptive_load_test(args))
    except KeyboardInterrupt:
        logger.info("Load test interrupted by user")
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
