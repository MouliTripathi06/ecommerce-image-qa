"""
OpenEnv typed models for the E-commerce Image QA environment.
All models use Pydantic for validation and serialization.
Compatible with Python 3.9+
"""
from typing import List, Optional, Dict, Any
# Literal is available in typing from Python 3.8+
from typing import Literal
from pydantic import BaseModel, Field


# ── Observation ──────────────────────────────────────────────────────────────

class ImageMetadata(BaseModel):
    width: int
    height: int
    channels: int = 3
    defect_count_hint: Optional[int] = None  # provided only in easy task


class Observation(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG product image")
    task_id: str  # one of: task_easy, task_medium, task_hard
    metadata: ImageMetadata
    step_number: int = Field(default=0, ge=0)
    episode_id: str


# ── Action ───────────────────────────────────────────────────────────────────

# Valid defect labels (used for documentation; validation done in graders)
VALID_DEFECTS = ("blur", "dark", "watermark", "wrong_background", "overexposed")
VALID_RECOMMENDATIONS = ("approve", "reject", "retouch")

class Action(BaseModel):
    """
    Agent's response for the current observation.

    - task_easy:   fill `defects` with exactly ONE label (or empty list = "clean")
    - task_medium: fill `defects` with all detected labels (0-5 items)
    - task_hard:   fill all three fields
    """
    defects: List[str] = Field(
        default_factory=list,
        description="Defect labels: blur | dark | watermark | wrong_background | overexposed",
    )
    severity: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Estimated severity 0.0-1.0 (required for task_hard)",
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Final quality decision: approve | reject | retouch (required for task_hard)",
    )


# ── Reward ───────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Reward signal 0.0-1.0")
    components: Dict[str, Any] = Field(
        default_factory=dict,
        description="Breakdown of reward sub-components",
    )
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ── State ────────────────────────────────────────────────────────────────────

class EnvState(BaseModel):
    episode_id: str
    task_id: str
    step_number: int
    total_reward: float
    done: bool
    ground_truth: Optional[Dict[str, Any]] = None  # revealed only after episode ends