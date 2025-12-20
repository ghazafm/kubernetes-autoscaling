from typing import Optional, Tuple, Union

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback


class HeuristicDQN(DQN):
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
        current_action = observation[0]
        cpu_rel = observation[1]
        mem_rel = observation[2]
        cpu_dist = observation[3]
        mem_dist = observation[4]
        response_time = observation[5]

        current_replica_action = int(current_action * 99)

        RT_HIGH = 0.8
        RT_CRITICAL = 1.0
        CPU_HIGH = 0.75
        MEM_HIGH = 0.75
        UNDERUTIL_THRESHOLD = -0.1

        adjustment = 0

        # RT Critical, jadi scale up banyak
        if response_time > RT_CRITICAL:
            self.custom_logger.debug(
                "Heuristic: Response Time Critical - Scaling Up Aggressively"
            )
            adjustment = np.random.randint(15, 25)

        # RT High, jadi scale up cukup banyak
        elif response_time > RT_HIGH:
            self.custom_logger.debug(
                "Heuristic: Response Time High - Scaling Up Moderately"
            )
            adjustment = np.random.randint(8, 15)

        # CPU or Memory High, scale up sedang
        elif cpu_rel > CPU_HIGH or mem_rel > MEM_HIGH:
            self.custom_logger.debug(
                "Heuristic: CPU or Memory High - Scaling Up Slightly"
            )
            adjustment = np.random.randint(5, 12)

        # CPU dan Memory Underutilized, scale down
        elif cpu_dist < UNDERUTIL_THRESHOLD and mem_dist < UNDERUTIL_THRESHOLD:
            underutil_severity = min(abs(cpu_dist), abs(mem_dist))
            if underutil_severity > 0.3:  # noqa: PLR2004
                adjustment = -np.random.randint(8, 15)
            elif underutil_severity > 0.2:  # noqa: PLR2004
                adjustment = -np.random.randint(5, 10)
            else:
                adjustment = -np.random.randint(2, 6)

        # Salah satu underutilized, scale down sedikit
        elif cpu_dist < UNDERUTIL_THRESHOLD or mem_dist < UNDERUTIL_THRESHOLD:
            self.custom_logger.debug(
                "Heuristic: CPU or Memory Underutilized - Scaling Down Slightly"
            )
            adjustment = -np.random.randint(2, 6)

        # Normal case, cenderung tidak berubah
        elif np.random.random() < 0.7:  # noqa: PLR2004
            self.custom_logger.debug("Heuristic: Normal Conditions - No Scaling")
            adjustment = 0

        else:
            self.custom_logger.debug("Heuristic: Normal Conditions - Minor Adjustment")
            adjustment = np.random.randint(-2, 3)

        new_action = current_replica_action + adjustment

        new_action = np.clip(new_action, 0, 99)

        return int(new_action)

    def predict(
        self,
        observation: Union[np.ndarray, dict],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if deterministic:
            # Untuk production
            return super().predict(observation, state, episode_start, deterministic)

        # Masih memungkinkan untuk random dengan probabilitas tertentu
        self.custom_logger.debug(f"Heuristic Probability: {self.heuristic_prob}")
        if np.random.random() < self.heuristic_prob:
            self.custom_logger.debug("Using heuristic action")
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

            self.custom_logger.debug(f"Heuristic Action(s): {actions}")

            return actions, state
        self.total_actions += 1
        return super().predict(observation, state, episode_start, deterministic)

    def decay_heuristic_prob(self):
        self.heuristic_prob = max(
            self.min_heuristic_prob, self.heuristic_prob * self.heuristic_decay
        )

    def get_heuristic_stats(self) -> dict:
        if self.total_actions == 0:
            usage_rate = 0.0
        else:
            usage_rate = self.heuristic_count / self.total_actions

        return {
            "current_heuristic_prob": self.heuristic_prob,
            "heuristic_count": self.heuristic_count,
            "total_actions": self.total_actions,
            "actual_usage_rate": usage_rate,
        }

    def reset_heuristic_stats(self):
        self.heuristic_count = 0
        self.total_actions = 0


class HeuristicDecayCallback(BaseCallback):
    def __init__(self, logger, verbose: int = 0):
        super().__init__(verbose)
        self.custom_logger = logger
        self.episode_count = 0

    def _on_step(self) -> bool:
        if self.locals.get("dones", [False])[0]:
            self.episode_count += 1

            if hasattr(self.model, "decay_heuristic_prob"):
                self.model.decay_heuristic_prob()

                if self.episode_count % 2 == 0:
                    stats = self.model.get_heuristic_stats()
                    if self.verbose > 0:
                        self.custom_logger.info(
                            f"Episode {self.episode_count}: "
                            f"Heuristic Prob: {stats['current_heuristic_prob']:.4f}, "
                            f"Heuristic Actions: {stats['heuristic_count']}, "
                            f"Total Actions: {stats['total_actions']}, "
                            f"Actual Usage Rate: {stats['actual_usage_rate']:.4f}"
                        )

                    self.model.reset_heuristic_stats()

        return True
