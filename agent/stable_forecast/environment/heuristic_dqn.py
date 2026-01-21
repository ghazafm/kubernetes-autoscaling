from typing import Optional, Tuple, Union

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


class HeuristicDQN(DQN):
    """
    DQN with heuristic-guided exploration for Kubernetes autoscaling.

    Updated for NEW observation format:
    - observation[0]: action (normalized 0-1)
    - observation[1]: cpu (0-1 range, where 0.5 = 50% CPU)
    - observation[2]: memory (0-1 range, where 0.5 = 50% memory)
    - observation[3]: response_time (0-3 range, where 1.0 = 100% of SLO)
    """

    def __init__(
        self,
        *args,
        custom_logger=None,
        heuristic_prob: float = 0.5,
        heuristic_decay: float = 0.995,
        min_heuristic_prob: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.heuristic_prob = heuristic_prob
        self.initial_heuristic_prob = heuristic_prob
        self.heuristic_decay = heuristic_decay
        self.min_heuristic_prob = min_heuristic_prob
        self.heuristic_count = 0
        self.total_actions = 0
        self.custom_logger = custom_logger

    def heuristic_action(self, observation: np.ndarray) -> int:
        """
        Simple RT-focused heuristic with trend awareness.

        Observation structure:
        - observation[0]: current action (0-1)
        - observation[3]: Response time (0-3, normalized by max_response_time/100)
        - observation[6]: Delta response time (trend)
        """
        current_action = observation[0]
        response_time = observation[3]
        delta_rt = observation[6]

        current_replica_action = int(current_action * 99)

        # Thresholds
        RT_CRITICAL = 1.0
        RT_HIGH = 0.8
        RT_LOW = 0.4
        TREND_THRESHOLD = 0.05

        adjustment = 0

        if response_time > RT_CRITICAL:
            adjustment = np.random.randint(15, 25)
            self.custom_logger.debug(
                f"CRITICAL: RT={response_time:.2f} - Scale up {adjustment}"
            )

        elif response_time > RT_HIGH:
            if delta_rt > TREND_THRESHOLD:
                adjustment = np.random.randint(12, 18)
                self.custom_logger.debug(
                    f"HIGH (trending up): RT={response_time:.2f}, Δ={delta_rt:.3f} "
                    f"- Scale up {adjustment}"
                )
            else:
                adjustment = np.random.randint(8, 15)
                self.custom_logger.debug(
                    f"HIGH: RT={response_time:.2f} - Scale up {adjustment}"
                )

        # Low: Can scale down
        elif response_time < RT_LOW:
            if delta_rt < -TREND_THRESHOLD:
                adjustment = -np.random.randint(8, 15)
                self.custom_logger.debug(
                    f"LOW (trending down): RT={response_time:.2f}, Δ={delta_rt:.3f} "
                    f"- Scale down {adjustment}"
                )
            else:
                adjustment = -np.random.randint(5, 12)
                self.custom_logger.debug(
                    f"LOW: RT={response_time:.2f} - Scale down {adjustment}"
                )

        elif delta_rt > TREND_THRESHOLD:
            adjustment = np.random.randint(3, 8)
            self.custom_logger.debug(
                f"NORMAL (bad trend): RT={response_time:.2f}, Δ={delta_rt:.3f} "
                f"- Preemptive scale up {adjustment}"
            )
        else:
            adjustment = 0
            self.custom_logger.debug(f"NORMAL: RT={response_time:.2f} - Maintain")

        new_action = current_replica_action + adjustment
        new_action = np.clip(new_action, 0, 99)

        self.custom_logger.debug(
            f"Heuristic: [{current_replica_action}] + {adjustment:+d} → [{new_action}]"
        )

        return int(new_action)

    def predict(
        self,
        observation: Union[np.ndarray, dict],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Predict action with optional heuristic guidance.

        When deterministic=True (production), always uses learned policy.
        When deterministic=False (training),
        uses heuristic with probability heuristic_prob.
        """
        if deterministic:
            # Production mode: always use learned policy
            return super().predict(observation, state, episode_start, deterministic)

        # Training mode: mix heuristic and learned policy
        if np.random.random() < self.heuristic_prob:
            self.custom_logger.debug(
                f"Using heuristic action (prob={self.heuristic_prob:.4f})"
            )

            if isinstance(observation, dict):
                obs = observation["observation"]
            else:
                obs = observation

            if len(obs.shape) > 1:
                actions = np.array([self.heuristic_action(o) for o in obs])
            else:
                actions = np.array([self.heuristic_action(obs)])

            self.heuristic_count += 1
            self.total_actions += 1

            return actions, state

        # Use learned policy
        self.custom_logger.debug("Using learned policy action")
        self.total_actions += 1
        return super().predict(observation, state, episode_start, deterministic)

    def decay_heuristic_prob(self):
        """Decay the heuristic probability after each episode."""
        old_prob = self.heuristic_prob
        self.heuristic_prob = max(
            self.min_heuristic_prob, self.heuristic_prob * self.heuristic_decay
        )

        if self.custom_logger and old_prob != self.heuristic_prob:
            self.custom_logger.debug(
                f"Heuristic probability decayed: {old_prob:.4f} → "
                f"{self.heuristic_prob:.4f}"
            )

    def get_heuristic_stats(self) -> dict:
        """Get statistics about heuristic usage."""
        if self.total_actions == 0:
            usage_rate = 0.0
        else:
            usage_rate = self.heuristic_count / self.total_actions

        return {
            "current_heuristic_prob": self.heuristic_prob,
            "heuristic_count": self.heuristic_count,
            "total_actions": self.total_actions,
            "actual_usage_rate": usage_rate,
            "initial_heuristic_prob": self.initial_heuristic_prob,
            "min_heuristic_prob": self.min_heuristic_prob,
            "heuristic_decay": self.heuristic_decay,
        }

    def reset_heuristic_stats(self):
        """Reset action counters (called periodically during training)."""
        self.heuristic_count = 0
        self.total_actions = 0


class HeuristicDecayCallback(BaseCallback):
    def __init__(self, logger, verbose: int = 0, log_every_n_episodes: int = 2):
        super().__init__(verbose)
        self.custom_logger = logger
        self.episode_count = 0
        self.log_every_n_episodes = log_every_n_episodes

    def _on_step(self) -> bool:
        """Called after each environment step."""
        # Check if episode ended
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1

            # Decay heuristic probability
            if hasattr(self.model, "decay_heuristic_prob"):
                self.model.decay_heuristic_prob()

                # Log statistics periodically
                if self.episode_count % self.log_every_n_episodes == 0:
                    stats = self.model.get_heuristic_stats()

                    if self.verbose > 0:
                        self.custom_logger.info(
                            f"Episode {self.episode_count} | "
                            f"Heuristic Stats: "
                            f"prob={stats['current_heuristic_prob']:.4f}, "
                            f"used={stats['heuristic_count']}"
                            f"/{stats['total_actions']}, "
                            f"rate={stats['actual_usage_rate']:.2%}"
                        )

                    self.model.reset_heuristic_stats()

        return True
