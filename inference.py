"""
inference.py — Baseline inference script for E-commerce Image QA OpenEnv.

Uses OpenAI client to call the configured LLM against the environment.
Emits structured stdout logs in [START] / [STEP] / [END] format as required.

Environment variables required:
  API_BASE_URL  — LLM API endpoint (e.g. https://api.openai.com/v1)
  MODEL_NAME    — Model identifier (e.g. gpt-4o)
  HF_TOKEN      — API key / HuggingFace token
"""
import os
import json
import base64
import sys
import time
import requests
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o")
HF_TOKEN     = os.environ.get("HF_TOKEN", "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")
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:8080")

TASKS = ["task_easy", "task_medium", "task_hard"]
EPISODES_PER_TASK = 5   # run 5 episodes per task for a stable baseline
SEED_BASE = 42

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ── Prompt builders ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert e-commerce product image quality reviewer.
Your job is to inspect product images and detect quality defects.

Possible defect labels (use ONLY these exact strings):
  blur            — image is blurry or out of focus
  dark            — image is too dark / underexposed
  watermark       — visible watermark or copyright text on the image
  wrong_background — background is not clean white/light-grey (cluttered, colored, or dark)
  overexposed     — image is too bright / washed out

You MUST respond with ONLY a valid JSON object. No markdown, no explanation.

For task_easy (single defect hint given):
  {"defects": ["<label>"] }   or   {"defects": []}  if clean

For task_medium (find all defects):
  {"defects": ["<label1>", "<label2>", ...]}

For task_hard (full review):
  {
    "defects": ["<label1>", ...],
    "severity": <float 0.0-1.0>,
    "recommendation": "<approve|reject|retouch>"
  }
"""

def build_user_message(obs: dict) -> list:
    """Build the multimodal user message from an observation."""
    task_id = obs["task_id"]
    hint = obs["metadata"].get("defect_count_hint")

    if task_id == "task_easy":
        instruction = f"Inspect this product image. There are exactly {hint} defect(s). Identify it (or confirm clean)."
    elif task_id == "task_medium":
        instruction = "Inspect this product image. List ALL quality defects present."
    else:
        instruction = (
            "Inspect this product image. List all defects, estimate severity (0.0–1.0), "
            "and give a recommendation (approve / reject / retouch)."
        )

    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{obs['image_base64']}",
                "detail": "low",
            },
        },
        {"type": "text", "text": instruction},
    ]


# ── Environment helpers ───────────────────────────────────────────────────────

def env_reset(task_id: str, seed: int) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(task_id: str, action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"task_id": task_id, "action": action}, timeout=30)
    r.raise_for_status()
    return r.json()


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(obs: dict) -> dict:
    """Call the LLM with the observation image and return parsed action dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(obs)},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=200,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if any
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: empty clean prediction
        action = {"defects": []}

    # Ensure required fields
    action.setdefault("defects", [])
    return action


# ── Main inference loop ───────────────────────────────────────────────────────

def run_task(task_id: str) -> float:
    """Run EPISODES_PER_TASK episodes on a task, return mean reward."""
    rewards = []

    for ep in range(EPISODES_PER_TASK):
        seed = SEED_BASE + ep

        # Reset environment
        obs = env_reset(task_id, seed)

        # ── [START] log ──────────────────────────────────────────────────────
        print(json.dumps({
            "event": "START",
            "task_id": task_id,
            "episode": ep,
            "episode_id": obs["episode_id"],
            "seed": seed,
        }))
        sys.stdout.flush()

        # Call LLM
        action = call_llm(obs)

        # Step the environment
        result = env_step(task_id, action)
        reward_value = result["reward"]["value"]
        rewards.append(reward_value)

        # ── [STEP] log ───────────────────────────────────────────────────────
        print(json.dumps({
            "event": "STEP",
            "task_id": task_id,
            "episode": ep,
            "episode_id": obs["episode_id"],
            "step": 1,
            "action": action,
            "reward": reward_value,
            "done": result["done"],
            "ground_truth_defects": result["info"].get("ground_truth_defects"),
            "ground_truth_severity": result["info"].get("ground_truth_severity"),
            "ground_truth_recommendation": result["info"].get("ground_truth_recommendation"),
            "reward_components": result["reward"].get("components", {}),
        }))
        sys.stdout.flush()

    mean_reward = round(sum(rewards) / len(rewards), 4)

    # ── [END] log ────────────────────────────────────────────────────────────
    print(json.dumps({
        "event": "END",
        "task_id": task_id,
        "episodes": EPISODES_PER_TASK,
        "rewards": rewards,
        "mean_reward": mean_reward,
    }))
    sys.stdout.flush()

    return mean_reward


def main():
    print(json.dumps({
        "event": "INFERENCE_START",
        "model": MODEL_NAME,
        "env_url": ENV_URL,
        "tasks": TASKS,
        "episodes_per_task": EPISODES_PER_TASK,
    }))
    sys.stdout.flush()

    results = {}
    for task_id in TASKS:
        mean_reward = run_task(task_id)
        results[task_id] = mean_reward

    overall = round(sum(results.values()) / len(results), 4)

    print(json.dumps({
        "event": "INFERENCE_COMPLETE",
        "results_per_task": results,
        "overall_mean_reward": overall,
    }))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
