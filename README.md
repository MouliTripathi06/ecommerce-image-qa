# 🛒 E-commerce Product Image QA — OpenEnv

> An OpenEnv environment where AI agents learn to do what human product photo reviewers do at Amazon, Flipkart, and Myntra — inspect images and decide if they're good enough to publish.

---

## Why This Exists

Every product listing on a major e-commerce platform needs its photos reviewed before going live. Is it blurry? Too dark? Does it have a watermark? Wrong background? Millions of images are reviewed by humans every day.

This environment simulates that exact task with **synthetically generated defects** — we *apply* the blur, we *stamp* the watermark — so the ground truth is always 100% known and the grader is always fair and deterministic.

---

## Action & Observation Spaces

### Observation
| Field | Type | Description |
|---|---|---|
| `image_base64` | `str` | Base64-encoded JPEG product image (256×256) |
| `task_id` | `str` | Which task is active |
| `metadata.width` | `int` | Image width (256) |
| `metadata.height` | `int` | Image height (256) |
| `metadata.defect_count_hint` | `int\|null` | Number of defects (only provided in task_easy) |
| `episode_id` | `str` | Unique ID for this episode |
| `step_number` | `int` | Current step |

### Action
| Field | Type | Required for | Description |
|---|---|---|---|
| `defects` | `list[str]` | All tasks | Labels from: `blur`, `dark`, `watermark`, `wrong_background`, `overexposed` |
| `severity` | `float` (0–1) | task_hard | Estimated defect severity |
| `recommendation` | `str` | task_hard | One of: `approve`, `reject`, `retouch` |

---

## Tasks

### 🟢 task_easy — Single Defect Detection
- The image has 0 or 1 defect.
- A hint tells the agent how many defects exist.
- Score: binary (correct = 1.0, partial = 0.5 for close misses, wrong = 0.0)

### 🟡 task_medium — Multi-Defect Classification
- The image may have 0–3 defects simultaneously.
- No hints provided.
- Score: **F1-score** over the defect label set (0.0–1.0)

### 🔴 task_hard — Full Quality Review
- The image has 1–4 defects (always at least one).
- Agent must detect all defects **+** estimate severity **+** recommend approve/reject/retouch.
- Score: weighted blend — 40% F1 + 30% severity accuracy + 30% recommendation accuracy

---

## Reward Function

Rewards provide **partial credit** throughout the trajectory:

- **Correct clean image**: 1.0
- **Exact defect match**: 1.0
- **Partial label overlap**: F1 proportional score
- **Severity within 0.1**: ~0.9 score
- **Correct recommendation**: full weight

No binary end-of-episode sparse reward — every episode gives a meaningful 0.0–1.0 signal.

---

## Setup & Usage

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the environment server
uvicorn app:app --reload --port 7860

# 3. Run baseline inference (in another terminal)
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export HF_TOKEN=your_api_key_here
export ENV_URL=http://localhost:7860

python inference.py
```

### Docker

```bash
docker build -t ecommerce-image-qa .
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o \
  -e HF_TOKEN=your_key \
  ecommerce-image-qa
```

### API Quick-start

```bash
# Reset
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy", "seed": 42}'

# Step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy", "action": {"defects": ["blur"]}}'

# State
curl http://localhost:7860/state?task_id=task_easy
```

---

## Baseline Scores

| Task | Mean Reward (5 episodes, gpt-4o, seed 42–46) |
|---|---|
| task_easy | ~0.75 |
| task_medium | ~0.62 |
| task_hard | ~0.51 |

---

## Project Structure

```
ecommerce-image-qa/
├── app.py                  # FastAPI server (OpenEnv HTTP endpoints)
├── inference.py            # Baseline inference script (REQUIRED)
├── openenv.yaml            # OpenEnv metadata spec
├── requirements.txt
├── Dockerfile
├── README.md
├── env/
│   ├── environment.py      # Core environment: reset/step/state
│   ├── image_generator.py  # Synthetic image generation + defect application
│   └── models.py           # Pydantic typed models (Observation/Action/Reward)
└── tasks/
    └── graders.py          # Deterministic graders for all 3 tasks
```
