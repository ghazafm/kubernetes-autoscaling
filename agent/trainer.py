import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Optional

from environment import KubernetesEnv
from rl import Q
from utils import log_verbose_details


@dataclass
class SaveConfig:
    note: str
    start_time: int


class Trainer:
    def __init__(
        self,
        agent: Q,
        env: KubernetesEnv,
        logger: Optional[logging.Logger] = None,
        resume: bool = False,
        resume_path: str = "",
        reset_epsilon: bool = True,
    ) -> None:
        self.agent = agent
        self.env = env
        self.logger = logger or logging.getLogger(__name__)
        self.savecfg: Optional[SaveConfig] = None
        self._old_sigint = None
        self._old_sigterm = None
        if resume and resume_path:
            try:
                start_epsilon = self.agent.epsilon
                self.agent.load_model(resume_path)
                self.logger.info(f"Resumed training from model at: {resume_path}")
                if reset_epsilon:
                    self.agent.epsilon = start_epsilon
                    self.logger.info(f"Epsilon reset to {self.agent.epsilon}")
            except Exception as e:
                self.logger.error(f"Failed to load model from {resume_path}: {e}")
                raise

    def _interrupted_save(self) -> Path | None:
        if not (self.agent and self.savecfg):
            self.logger.warning("⚠️ Nothing to save yet.")
            return None

        ext = (
            ".pth" if getattr(self.agent, "agent_type", "").upper() == "DQN" else ".pkl"
        )
        model_type = (
            "dqn"
            if getattr(self.agent, "agent_type", "").upper() == "DQN"
            else "qlearning"
        )
        interrupted_dir = Path(
            f"model/{model_type}/{self.savecfg.note}_{self.savecfg.start_time}/interrupted"
        )
        interrupted_dir.mkdir(parents=True, exist_ok=True)

        ts = int(time.time())
        ep = getattr(self.agent, "episodes_trained", 0)
        path = interrupted_dir / f"interrupted_episode_{ep}_{ts}{ext}"

        self.agent.save_model(str(path), ep)
        self.logger.info(f"✅ Model saved to: {path}")
        self.logger.info(f"Episodes completed: {ep}")
        return path

    def _signal_handler(self, signum: int, frame: FrameType) -> None:
        self.logger.warning(f"\nSignal {signum} received. Saving model...")
        try:
            self._interrupted_save()
        finally:
            signal.signal(signal.SIGINT, self._old_sigint or signal.SIG_DFL)
            signal.signal(signal.SIGTERM, self._old_sigterm or signal.SIG_DFL)
            # Re-emit to terminate (optional; comment out if you prefer to *not* exit)
            if signum == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(0)

    def _install_signal_handlers(self) -> None:
        self._old_sigint = signal.getsignal(signal.SIGINT)
        self._old_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _restore_signal_handlers(self) -> None:
        if self._old_sigint is not None:
            signal.signal(signal.SIGINT, self._old_sigint)
        if self._old_sigterm is not None:
            signal.signal(signal.SIGTERM, self._old_sigterm)

    def train(self, episodes: int, note: str, start_time: int) -> None:
        self.savecfg = SaveConfig(note=note, start_time=start_time)
        self._install_signal_handlers()
        try:
            total_best = float("-inf")
            for ep in range(episodes):
                self.agent.add_episode_count()
                self.logger.info(f"\nEpisode {ep + 1}/{episodes}")
                self.logger.info(
                    f"Total episodes trained: {self.agent.episodes_trained}"
                )
                self.logger.info(f"Epsilon: {self.agent.epsilon:.4f}")

                obs = self.env.reset()
                total = 0.0
                while True:
                    act = self.agent.get_action(obs)
                    nxt, rew, term, info = self.env.step(act)
                    self.agent.update_q_table(obs, act, rew, nxt)
                    total += rew
                    obs = nxt
                    self.logger.info(
                        f"Action: {act}, Reward: {rew}, Total: {total} | "
                        f"Iteration: {info['iteration']}"
                    )

                    # Debug: check observation type
                    self.logger.debug(f"Observation type: {type(obs)}, value: {obs}")

                    log_verbose_details(
                        observation=obs,
                        agent=self.agent,
                        verbose=True,
                        logger=self.logger,
                    )

                    if term:
                        self.agent.epsilon = max(
                            self.agent.epsilon_min,
                            self.agent.epsilon * self.agent.epsilon_decay,
                        )
                        break

                self.logger.info(f"Episode {ep + 1} completed. Total reward: {total}")
                if total > total_best:
                    total_best = total
                    self._save_checkpoint(ep, total_best, note, start_time)

        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user. Attempting to save...")
            self._interrupted_save()
            raise
        finally:
            self._restore_signal_handlers()
            self.logger.info("Training finished (cleanup done).")

    def _save_checkpoint(
        self, episode: int, score: float, note: str, start_time: int
    ) -> None:
        ext = (
            ".pth" if getattr(self.agent, "agent_type", "").upper() == "DQN" else ".pkl"
        )
        model_type = (
            "dqn"
            if getattr(self.agent, "agent_type", "").upper() == "DQN"
            else "qlearning"
        )
        path = (
            f"model/{model_type}/{note}_{start_time}/checkpoints/"
            f"episode_{episode}_total_{score}{ext}"
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save_model(path, episode + 1)
        self.logger.info(f"New best model saved with total reward: {score}")
