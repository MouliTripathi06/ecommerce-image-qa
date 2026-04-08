"""
ImageQAEnvironment — the core OpenEnv-compliant environment class.
Implements step(), reset(), state() per the OpenEnv specification.
Compatible with Python 3.9+
"""
import uuid
import random
from typing import Tuple, Dict, Any, List, Optional

from models import Observation, Action, Reward, EnvState, ImageMetadata
from image_generator import (
    generate_image_with_defects,
    compute_severity,
    compute_recommendation,
    DEFECT_TYPES,
)
from graders import grade_easy, grade_medium, grade_hard

MAX_STEPS_PER_EPISODE = 1  # Each episode = one image to review


class ImageQAEnvironment:
    """
    E-commerce product image quality review environment.

    Three tasks of increasing difficulty:
      task_easy   — detect a single defect (or confirm clean)
      task_medium — identify ALL defects present
      task_hard   — classify defects + rate severity + recommend action
    """

    def __init__(self, task_id: str = "task_medium", seed: int = None):
        assert task_id in ("task_easy", "task_medium", "task_hard"), \
            f"Unknown task_id: {task_id}"
        self.task_id = task_id
        self._seed = seed
        if seed is not None:
            random.seed(seed)

        # Episode state
        self._episode_id: str = ""
        self._step_number: int = 0
        self._done: bool = True
        self._total_reward: float = 0.0
        self._current_image_b64: str = ""
        self._ground_truth_defects: list = []
        self._ground_truth_severity: float = 0.0
        self._ground_truth_recommendation: str = "approve"
        self._defect_params: dict = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Start a new episode. Returns the initial observation."""
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_number = 0
        self._done = False
        self._total_reward = 0.0

        # Generate image with appropriate difficulty
        defects = self._sample_defects_for_task()
        b64, applied_defects, meta = generate_image_with_defects(defects=defects)

        self._current_image_b64 = b64
        self._ground_truth_defects = applied_defects
        self._defect_params = meta["params"]
        self._ground_truth_severity = compute_severity(applied_defects, self._defect_params)
        self._ground_truth_recommendation = compute_recommendation(
            applied_defects, self._ground_truth_severity
        )

        obs = Observation(
            image_base64=b64,
            task_id=self.task_id,
            metadata=ImageMetadata(
                width=256,
                height=256,
                channels=3,
                # Hint for easy task only: tell agent how many defects exist
                defect_count_hint=len(applied_defects) if self.task_id == "task_easy" else None,
            ),
            step_number=0,
            episode_id=self._episode_id,
        )
        return obs

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Process agent action, compute reward, return (obs, reward, done, info).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_number += 1
        self._done = True  # single-step episodes

        # Score the action
        reward_value, components = self._score_action(action)
        self._total_reward = reward_value

        reward = Reward(
            value=reward_value,
            components=components,
            done=True,
            info={
                "episode_id": self._episode_id,
                "step": self._step_number,
                "ground_truth_defects": self._ground_truth_defects,
                "ground_truth_severity": self._ground_truth_severity,
                "ground_truth_recommendation": self._ground_truth_recommendation,
            },
        )

        # Return same obs (episode over) and reveal ground truth
        obs = Observation(
            image_base64=self._current_image_b64,
            task_id=self.task_id,
            metadata=ImageMetadata(width=256, height=256, channels=3),
            step_number=self._step_number,
            episode_id=self._episode_id,
        )

        return obs, reward, True, reward.info

    def state(self) -> EnvState:
        """Return current environment state."""
        gt = None
        if self._done:
            gt = {
                "defects": self._ground_truth_defects,
                "severity": self._ground_truth_severity,
                "recommendation": self._ground_truth_recommendation,
            }
        return EnvState(
            episode_id=self._episode_id,
            task_id=self.task_id,
            step_number=self._step_number,
            total_reward=self._total_reward,
            done=self._done,
            ground_truth=gt,
        )

    # ── Internals ───────────────────────────────────────────────────────────

    def _sample_defects_for_task(self) -> list:
        """Sample defects appropriate for the task difficulty."""
        if self.task_id == "task_easy":
            # 0 or 1 defect, uniform
            if random.random() < 0.3:
                return []
            return [random.choice(DEFECT_TYPES)]
        elif self.task_id == "task_medium":
            # 0–3 defects
            num = random.choices([0, 1, 2, 3], weights=[10, 45, 30, 15])[0]
            return random.sample(DEFECT_TYPES, k=num)
        else:  # hard
            # 1–4 defects, always at least one
            num = random.choices([1, 2, 3, 4], weights=[20, 40, 30, 10])[0]
            return random.sample(DEFECT_TYPES, k=num)

    def _score_action(self, action: Action) -> Tuple[float, dict]:
        if self.task_id == "task_easy":
            return grade_easy(action.defects, self._ground_truth_defects)
        elif self.task_id == "task_medium":
            return grade_medium(action.defects, self._ground_truth_defects)
        else:
            return grade_hard(
                action.defects,
                self._ground_truth_defects,
                action.severity,
                self._ground_truth_severity,
                action.recommendation,
                self._ground_truth_recommendation,
            )
