"""
FastAPI application — OpenEnv HTTP interface.

Endpoints:
- GET  /         -> Simple browser UI
- GET  /health   -> Health check
- POST /reset    -> Observation
- POST /step     -> {observation, reward, done, info}
- GET  /state    -> EnvState

Accepts BOTH request formats for /step:
Nested: {"task_id": "task_easy", "action": {"defects": ["blur"]}}
Flat:   {"task_id": "task_easy", "defects": ["blur"]}
"""

from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

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


def _state_to_dict(state_obj) -> dict:
    return {
        "episode_id": state_obj.episode_id,
        "task_id": state_obj.task_id,
        "step_number": state_obj.step_number,
        "total_reward": state_obj.total_reward,
        "done": state_obj.done,
        "ground_truth": state_obj.ground_truth,
    }


class ResetRequest(BaseModel):
    task_id: str = "task_medium"
    seed: Optional[int] = 42


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>E-commerce Image QA — OpenEnv</title>
  <style>
    :root {
      --bg: #0b1020;
      --panel: #121933;
      --panel-2: #1a2345;
      --text: #e9eeff;
      --muted: #aab6e8;
      --accent: #6ea8fe;
      --accent-2: #7ef0c6;
      --danger: #ff8a8a;
      --border: #2b386b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, Arial, sans-serif;
      background: linear-gradient(180deg, #0b1020 0%, #0e1530 100%);
      color: var(--text);
    }
    .wrap {
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      margin-bottom: 20px;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 2rem;
    }
    p {
      color: var(--muted);
      line-height: 1.6;
    }
    .grid {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 20px;
    }
    .card {
      background: rgba(18, 25, 51, 0.95);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    label {
      display: block;
      margin: 12px 0 6px;
      font-weight: 600;
    }
    select, input, button, textarea {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      padding: 12px;
      font-size: 14px;
    }
    textarea {
      min-height: 120px;
      resize: vertical;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .checks {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 10px;
      background: var(--panel-2);
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .check input {
      width: auto;
      margin: 0;
    }
    .actions {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }
    .actions button {
      cursor: pointer;
      font-weight: 700;
    }
    .primary {
      background: var(--accent);
      color: #071224;
      border: none;
    }
    .secondary {
      background: transparent;
      color: var(--text);
    }
    .status {
      margin-top: 14px;
      font-size: 14px;
      color: var(--accent-2);
    }
    .error {
      color: var(--danger);
    }
    .preview {
      display: grid;
      gap: 16px;
    }
    .image-box {
      background: #0d142b;
      border: 1px solid var(--border);
      border-radius: 14px;
      min-height: 320px;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }
    .image-box img {
      max-width: 100%;
      height: auto;
      display: block;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #0d142b;
      padding: 14px;
      border-radius: 12px;
      border: 1px solid var(--border);
      color: #dce6ff;
      overflow: auto;
      max-height: 420px;
    }
    .meta {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }
    .pill {
      background: #0d142b;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
    }
    .pill small {
      display: block;
      color: var(--muted);
      margin-bottom: 6px;
    }
    a {
      color: var(--accent);
    }
    @media (max-width: 900px) {
      .grid {
        grid-template-columns: 1fr;
      }
      .meta {
        grid-template-columns: 1fr;
      }
      .row {
        grid-template-columns: 1fr;
      }
      .checks {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>🛒 E-commerce Image QA — OpenEnv</h1>
      <p>
        Inspect a synthetic product image, choose detected defects, and submit a review step.
        You can also open <a href="/docs" target="_blank" rel="noopener noreferrer">/docs</a>
        or test <a href="/health" target="_blank" rel="noopener noreferrer">/health</a>.
      </p>
    </div>

    <div class="grid">
      <div class="card">
        <label for="task_id">Task</label>
        <select id="task_id">
          <option value="task_easy">task_easy</option>
          <option value="task_medium" selected>task_medium</option>
          <option value="task_hard">task_hard</option>
        </select>

        <div class="row">
          <div>
            <label for="seed">Seed</label>
            <input id="seed" type="number" value="42" />
          </div>
          <div>
            <label for="severity">Severity (task_hard)</label>
            <input id="severity" type="number" min="0" max="1" step="0.01" placeholder="0.50" />
          </div>
        </div>

        <label for="recommendation">Recommendation (task_hard)</label>
        <select id="recommendation">
          <option value="">-- optional --</option>
          <option value="approve">approve</option>
          <option value="reject">reject</option>
          <option value="retouch">retouch</option>
        </select>

        <label>Defects</label>
        <div class="checks">
          <label class="check"><input type="checkbox" value="blur" /> blur</label>
          <label class="check"><input type="checkbox" value="dark" /> dark</label>
          <label class="check"><input type="checkbox" value="watermark" /> watermark</label>
          <label class="check"><input type="checkbox" value="wrong_background" /> wrong_background</label>
          <label class="check"><input type="checkbox" value="overexposed" /> overexposed</label>
        </div>

        <div class="actions">
          <button class="primary" onclick="resetEnv()">Reset</button>
          <button class="secondary" onclick="submitStep()">Submit Step</button>
          <button class="secondary" onclick="loadState()">Load State</button>
        </div>

        <div id="status" class="status">Ready.</div>
      </div>

      <div class="preview">
        <div class="card">
          <div class="image-box">
            <img id="previewImage" alt="Observation preview" />
          </div>
        </div>

        <div class="meta">
          <div class="pill"><small>Episode ID</small><div id="episodeId">—</div></div>
          <div class="pill"><small>Step</small><div id="stepNumber">—</div></div>
          <div class="pill"><small>Task</small><div id="taskLabel">—</div></div>
        </div>

        <div class="card">
          <h3 style="margin-top:0;">Latest Response</h3>
          <pre id="responseBox">{}</pre>
        </div>
      </div>
    </div>
  </div>

  <script>
    function selectedDefects() {
      return Array.from(document.querySelectorAll('.checks input:checked')).map(el => el.value);
    }

    function setStatus(message, isError = false) {
      const el = document.getElementById('status');
      el.textContent = message;
      el.className = isError ? 'status error' : 'status';
    }

    function updateObservation(obs) {
      if (!obs) return;
      document.getElementById('episodeId').textContent = obs.episode_id ?? '—';
      document.getElementById('stepNumber').textContent = obs.step_number ?? '—';
      document.getElementById('taskLabel').textContent = obs.task_id ?? '—';

      if (obs.image_base64) {
        const img = document.getElementById('previewImage');
        img.src = 'data:image/jpeg;base64,' + obs.image_base64;
      }
    }

    function showJson(data) {
      document.getElementById('responseBox').textContent = JSON.stringify(data, null, 2);
    }

    async function resetEnv() {
      try {
        setStatus('Resetting environment...');
        const task_id = document.getElementById('task_id').value;
        const seed = parseInt(document.getElementById('seed').value || '42', 10);

        const res = await fetch('/reset', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ task_id, seed })
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Reset failed');

        updateObservation(data);
        showJson(data);
        setStatus('Environment reset successfully.');
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    async function submitStep() {
      try {
        setStatus('Submitting step...');
        const task_id = document.getElementById('task_id').value;
        const defects = selectedDefects();
        const severityRaw = document.getElementById('severity').value.trim();
        const recommendation = document.getElementById('recommendation').value.trim();

        const payload = {
          task_id,
          action: {
            defects
          }
        };

        if (severityRaw !== '') {
          payload.action.severity = parseFloat(severityRaw);
        }
        if (recommendation !== '') {
          payload.action.recommendation = recommendation;
        }

        const res = await fetch('/step', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Step failed');

        if (data.observation) {
          updateObservation(data.observation);
        }
        showJson(data);
        setStatus('Step submitted successfully.');
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    async function loadState() {
      try {
        setStatus('Loading state...');
        const task_id = document.getElementById('task_id').value;
        const res = await fetch('/state?task_id=' + encodeURIComponent(task_id));
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'State load failed');
        showJson(data);
        setStatus('State loaded.');
      } catch (err) {
        setStatus(err.message, true);
      }
    }

    resetEnv();
  </script>
</body>
</html>
    """


@app.get("/health")
def health():
    return {"status": "ok", "service": "ecommerce-image-qa-openenv"}


@app.post("/reset")
def reset(request: ResetRequest):
    try:
        task_id = request.task_id or "task_easy"
        seed = request.seed if request.seed is not None else 42

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

        if "action" in body:
            action_data = body["action"]
        else:
            action_data = body

        defects = action_data.get("defects", [])
        severity = action_data.get("severity")
        recommendation = action_data.get("recommendation")

        action = Action(
            defects=defects,
            severity=severity,
            recommendation=recommendation,
        )

        env = _get_env(task_id)
        obs2, reward, done, info = env.step(action)

        return JSONResponse(
            content={
                "observation": _obs_to_dict(obs2),
                "reward": _reward_to_dict(reward),
                "done": done,
                "info": info,
            }
        )

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=500,
            detail=str(e) + "\\n" + traceback.format_exc()
        )


@app.get("/state")
def state(task_id: str = Query(default="task_medium")):
    try:
        env = _get_env(task_id)
        return JSONResponse(content=_state_to_dict(env.state()))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def server():
    return app
