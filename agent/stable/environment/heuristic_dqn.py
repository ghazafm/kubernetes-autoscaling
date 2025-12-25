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
        Heuristic-based action selection using NEW observation format.

        Observation structure:
        - observation[0]: current action (0-1)
        - observation[1]: CPU utilization (0-1)
        - observation[2]: Memory utilization (0-1)
        - observation[3]: Response time (0-3, normalized by max_response_time/100)
        """
        current_action = observation[0]
        cpu = observation[1]  # 0-1 range (0.5 = 50% CPU)
        memory = observation[2]  # 0-1 range (0.5 = 50% memory)
        response_time = observation[3]  # 0-3 range (1.0 = 100% of SLO)

        current_replica_action = int(current_action * 99)

        # Thresholds for decision making
        RT_HIGH = 0.8  # 80% of max response time
        RT_CRITICAL = 1.0  # 100% of max response time (SLO violation)

        CPU_HIGH = 0.7  # 70% CPU utilization
        CPU_CRITICAL = 0.85  # 85% CPU utilization
        CPU_LOW = 0.3  # 30% CPU utilization
        CPU_VERY_LOW = 0.15  # 15% CPU utilization

        MEM_HIGH = 0.7  # 70% memory utilization
        MEM_CRITICAL = 0.85  # 85% memory utilization
        MEM_LOW = 0.3  # 30% memory utilization
        MEM_VERY_LOW = 0.15  # 15% memory utilization

        adjustment = 0

        # Priority 1: Response Time Critical - Scale up aggressively
        if response_time > RT_CRITICAL:
            self.custom_logger.debug(
                f"Heuristic: RT Critical ({response_time:.2f}) - "
                "Scaling Up Aggressively"
            )
            adjustment = np.random.randint(15, 25)

        # Priority 2: Response Time High - Scale up moderately
        elif response_time > RT_HIGH:
            self.custom_logger.debug(
                f"Heuristic: RT High ({response_time:.2f}) - Scaling Up Moderately"
            )
            adjustment = np.random.randint(8, 15)

        # Priority 3: CPU or Memory Critical - Scale up
        elif cpu > CPU_CRITICAL or memory > MEM_CRITICAL:
            self.custom_logger.debug(
                f"Heuristic: Resources Critical (CPU:{cpu:.2f}, "
                f"MEM:{memory:.2f}) - Scaling Up"
            )
            adjustment = np.random.randint(10, 18)

        # Priority 4: CPU or Memory High - Scale up slightly
        elif cpu > CPU_HIGH or memory > MEM_HIGH:
            self.custom_logger.debug(
                f"Heuristic: Resources High (CPU:{cpu:.2f}, "
                f"MEM:{memory:.2f}) - Scaling Up Slightly"
            )
            adjustment = np.random.randint(5, 12)

        # Priority 5: Both CPU and Memory Very Low - Scale down aggressively
        elif cpu < CPU_VERY_LOW and memory < MEM_VERY_LOW:
            self.custom_logger.debug(
                f"Heuristic: Resources Very Low (CPU:{cpu:.2f}, "
                f"MEM:{memory:.2f}) - Scaling Down Aggressively"
            )
            adjustment = -np.random.randint(8, 15)

        # Priority 6: Both CPU and Memory Low - Scale down moderately
        elif cpu < CPU_LOW and memory < MEM_LOW:
            self.custom_logger.debug(
                f"Heuristic: Resources Low (CPU:{cpu:.2f}, "
                f"MEM:{memory:.2f}) - Scaling Down Moderately"
            )
            adjustment = -np.random.randint(5, 10)

        # Priority 7: One resource is low - Scale down slightly
        elif cpu < CPU_LOW or memory < MEM_LOW:
            self.custom_logger.debug(
                f"Heuristic: One Resource Low (CPU:{cpu:.2f}, "
                f"MEM:{memory:.2f}) - Scaling Down Slightly"
            )
            adjustment = -np.random.randint(2, 6)

        # Priority 8: Normal conditions - Maintain or slight adjustment
        elif np.random.random() < 0.7:  # noqa: PLR2004
            self.custom_logger.debug(
                f"Heuristic: Normal Conditions (CPU:{cpu:.2f}, "
                f"MEM:{memory:.2f}, RT:{response_time:.2f}) - Maintaining"
            )
            adjustment = 0

        else:
            self.custom_logger.debug(
                "Heuristic: Normal Conditions - Minor Random Adjustment"
            )
            adjustment = np.random.randint(-2, 3)

        new_action = current_replica_action + adjustment
        new_action = np.clip(new_action, 0, 99)

        self.custom_logger.debug(
            f"Heuristic Decision: Current Action={current_replica_action}, "
            f"Adjustment={adjustment:+d}, New Action={new_action}"
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
