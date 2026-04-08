""""
FastAPI application — OpenEnv HTTP interface.
POST /reset  -> Observation
POST /step   -> {observation, reward, done, info}
GET  /state  -> EnvState
GET  /health -> 200 OK

Accepts BOTH request formats for /step:
  Nested:  {"task_id": "task_easy", "action": {"defects": ["blur"]}}
  Flat:    {"task_id": "task_easy", "defects": ["blur"]}
"""
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from environment import ImageQAEnvironment
from models import Action

app = FastAPI(title="E-commerce Image QA — OpenEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: Dict[str, ImageQAEnvironment] = {}


def _get_env(task_id: str) -> ImageQAEnvironment:
    if task_id not in _envs:
        env = ImageQAEnvironment(task_id=task_id)
        env.reset()
        _envs[task_id] = env
    return _envs[task_id]


def _obs_to_dict(obs) -> dict:
    return {
        "image_base64": obs.image_base64,
        "task_id": obs.task_id,
        "metadata": {
            "width": obs.metadata.width,
            "height": obs.metadata.height,
            "channels": obs.metadata.channels,
            "defect_count_hint": obs.metadata.defect_count_hint,
        },
        "step_number": obs.step_number,
        "episode_id": obs.episode_id,
    }


def _reward_to_dict(reward) -> dict:
    return {
        "value": reward.value,
        "components": reward.components,
        "done": reward.done,
        "info": reward.info,
    }


def _state_to_dict(s) -> dict:
    return {
        "episode_id": s.episode_id,
        "task_id": s.task_id,
        "step_number": s.step_number,
        "total_reward": s.total_reward,
        "done": s.done,
        "ground_truth": s.ground_truth,
    }


class ResetRequest(BaseModel):
    task_id: str = "task_medium"
    seed: Optional[int] = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "ecommerce-image-qa-openenv"}


@app.post("/reset")
def reset(request: dict = {}):
    try:
        task_id = request.get("task_id", "task_easy")
        seed = request.get("seed", 42)
        env = ImageQAEnvironment(task_id=task_id, seed=seed)
        obs = env.reset()
        _envs[task_id] = env
        return JSONResponse(content=_obs_to_dict(obs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: Request):
    """
    Accepts both formats:
      Nested: {"task_id": "task_easy", "action": {"defects": ["blur"]}}
      Flat:   {"task_id": "task_easy", "defects": ["blur"]}
    """
    try:
        body = await request.json()

        task_id = body.get("task_id", "task_medium")

        # Support both nested {"action": {...}} and flat {"defects": [...]}
        if "action" in body:
            action_data = body["action"]
        else:
            action_data = body  # flat format

        defects         = action_data.get("defects", [])
        severity        = action_data.get("severity", None)
        recommendation  = action_data.get("recommendation", None)

        action = Action(
            defects=defects,
            severity=severity,
            recommendation=recommendation,
        )

        env = _get_env(task_id)
        obs2, reward, done, info = env.step(action)

        return JSONResponse(content={
            "observation": _obs_to_dict(obs2),
            "reward": _reward_to_dict(reward),
            "done": done,
            "info": info,
        })

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=str(e) + "\n" + traceback.format_exc())


@app.get("/state")
def state(task_id: str = Query(default="task_medium")):
    env = _get_env(task_id)
    return JSONResponse(content=_state_to_dict(env.state()))

def server():
    return app
