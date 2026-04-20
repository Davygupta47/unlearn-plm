# CLAUDE.md — Adaptive Chatbot Full-Stack Website

## Project Identity

**Name:** `adaptive-chatbot-unlearn` (web platform)
**Goal:** A full-stack web application that exposes the adaptive-chatbot unlearning system — users chat with a live Qwen1.5-0.5B model, rate responses, trigger unlearning cycles, and watch the model improve in real time via a live metrics dashboard.
**Model backend:** Qwen1.5-0.5B + llm_unlearn pipeline (AscentPlusKLDivergence primary)
**Target infra:** Single T4 GPU VM (Colab Pro / Kaggle / GCP n1-standard-4 + T4)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          BROWSER (React)                            │
│  ┌───────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │  Chat UI      │  │  Metrics Dashboard│  │  Admin / Cycle Panel │ │
│  │  thumbs ±     │  │  PPL / MIA / AUC │  │  force unlearn       │ │
│  └──────┬────────┘  └────────┬─────────┘  └─────────┬────────────┘ │
└─────────┼───────────────────┼──────────────────────┼──────────────┘
          │  REST + SSE        │  REST (polling)        │  REST
          ▼                    ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FASTAPI  (Python 3.11)                        │
│                                                                     │
│  POST /chat          → inference worker → Qwen model               │
│  POST /feedback      → write forget/retain buffer + check trigger   │
│  GET  /metrics       → read cycle_history.json + current eval JSON  │
│  POST /admin/unlearn → manual cycle trigger (admin only)            │
│  GET  /stream/events → SSE: real-time cycle progress               │
│  GET  /health        → model loaded? cycle running?                 │
│                                                                     │
│  ┌──────────────────┐   ┌─────────────────────────────────────┐    │
│  │  InferenceWorker │   │  CycleWorker (BackgroundTask/thread)│    │
│  │  – loads model   │   │  – tokenize → unlearn → re-finetune │    │
│  │  – generate()    │   │  – eval gates → symlink update       │    │
│  │  – fp16 safe     │   │  – SSE progress events               │    │
│  └──────────────────┘   └─────────────────────────────────────┘    │
└───────────────────┬────────────────────┬────────────────────────────┘
                    │                    │
          ┌─────────▼──────┐   ┌────────▼───────────────────┐
          │   PostgreSQL   │   │   Redis (optional)          │
          │  (chat logs,   │   │  (SSE pub/sub, rate limits, │
          │   cycle meta,  │   │   session cache)            │
          │   user prefs)  │   └────────────────────────────┘
          └────────────────┘
                    │
          ┌─────────▼──────────────────────────────────┐
          │       Filesystem  (model weights + PT files)│
          │  ./models/Qwen1.5-0.5B/                     │
          │  ./output/chatbot/current  ← symlink         │
          │  ./data/feedback/forget_buffer.jsonl         │
          │  ./data/feedback/retain_buffer.jsonl         │
          │  ./tokenized_dataset/chatbot/               │
          │  ./output/logs/cycle_history.json           │
          └────────────────────────────────────────────┘
```

---

## Tech Stack

### Frontend
| Layer | Choice | Reason |
|---|---|---|
| Framework | **React 18 + Vite** | Fast HMR, minimal boilerplate |
| Language | **TypeScript** | Required for complex state |
| Styling | **Tailwind CSS v3** | Utility-first, consistent design tokens |
| Charts | **Recharts** | PPL/AUC trend lines, lightweight |
| State | **Zustand** | Simple global store (chat history, cycle state) |
| Data fetching | **TanStack Query v5** | Caching + auto-refetch for metrics |
| SSE client | **EventSource API** (native) | Cycle progress streaming |
| Auth | **Clerk** (or simple JWT) | User sessions for feedback attribution |
| Icons | **Lucide React** | Consistent icon set |
| Routing | **React Router v6** | SPA routes |

### Backend
| Layer | Choice | Reason |
|---|---|---|
| Framework | **FastAPI** | Async, OpenAPI auto-docs, Python native |
| Language | **Python 3.11** | Matches ML stack |
| ASGI server | **Uvicorn + Gunicorn** | Production multi-worker |
| Task queue | **Celery + Redis** (or `asyncio.BackgroundTasks`) | Unlearn cycle is long-running |
| Auth | **python-jose + passlib** | JWT tokens |
| Validation | **Pydantic v2** | Request/response schemas |
| ORM | **SQLAlchemy 2.0 async** | Async DB access |
| Migrations | **Alembic** | Schema versioning |

### Database
| Role | DB | Notes |
|---|---|---|
| Primary persistence | **PostgreSQL 15** | Chat logs, user data, cycle metadata |
| Cache / pub-sub | **Redis 7** | SSE event bus, session store, rate limit counters |
| File store | **Local filesystem** | Model weights, PT datasets, feedback JSONL (too large for DB) |

### Model Pipeline (existing `llm_unlearn` repo)
| Component | File | Notes |
|---|---|---|
| Inference | `chatbot_unlearn/serving/model_loader.py` | Loads `output/chatbot/current` |
| Feedback log | `chatbot_unlearn/pipeline/feedback_logger.py` | Writes JSONL buffers |
| Cycle manager | `chatbot_unlearn/pipeline/cycle_manager.py` | Full loop orchestrator |
| Unlearn | `chatbot_unlearn/method/akl.py` | AscentPlusKLDivergence (primary) |
| Eval | `chatbot_unlearn/evaluation/run_eval.py` | PPL + acc |
| MIA | `chatbot_unlearn/evaluation/run_mia.py` | AUC |

### DevOps / Infra
| Tool | Purpose |
|---|---|
| **Docker + Docker Compose** | Containerize API + Postgres + Redis |
| **Nginx** | Reverse proxy, static file serving, SSL termination |
| **GitHub Actions** | CI: lint, type-check, unit tests |
| **Fly.io / GCP / Vast.ai** | GPU-attached VM deployment |
| **Weights & Biases** | Experiment tracking (optional, offline fallback) |

---

## Repository Structure

```
adaptive-chatbot-web/
│
├── CLAUDE.md                          ← This file
├── README.md
├── docker-compose.yml                 ← Postgres + Redis + API + Frontend
├── .env.example
├── Makefile                           ← Common commands
│
├── frontend/                          ← React + Vite SPA
│   ├── index.html
│   ├── vite.config.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── package.json
│   └── src/
│       ├── main.tsx
│       ├── App.tsx                    ← Route layout
│       │
│       ├── pages/
│       │   ├── ChatPage.tsx           ← Main chat UI
│       │   ├── DashboardPage.tsx      ← Metrics + cycle history
│       │   ├── AdminPage.tsx          ← Manual cycle trigger + config
│       │   └── LoginPage.tsx
│       │
│       ├── components/
│       │   ├── chat/
│       │   │   ├── ChatWindow.tsx     ← Message list + scroll anchor
│       │   │   ├── MessageBubble.tsx  ← With thumbs-up/down buttons
│       │   │   ├── InputBar.tsx       ← Textarea + send button
│       │   │   └── ThumbsWidget.tsx   ← POST /feedback on click
│       │   │
│       │   ├── dashboard/
│       │   │   ├── MetricsCard.tsx    ← forget_ppl, retain_ppl, MIA_AUC
│       │   │   ├── PPLChart.tsx       ← Recharts line chart over cycles
│       │   │   ├── CycleTable.tsx     ← cycle_history.json rendered
│       │   │   └── CycleProgress.tsx  ← SSE-driven live progress bar
│       │   │
│       │   ├── admin/
│       │   │   ├── TriggerPanel.tsx   ← Button: POST /admin/unlearn
│       │   │   ├── ConfigEditor.tsx   ← Edit FORGET_TRIGGER, lr, etc.
│       │   │   └── BufferStatus.tsx   ← forget/retain buffer counts
│       │   │
│       │   └── shared/
│       │       ├── Navbar.tsx
│       │       ├── Badge.tsx          ← cycle number, method label
│       │       └── StreamBanner.tsx   ← SSE connection status
│       │
│       ├── store/
│       │   ├── chatStore.ts           ← Zustand: messages, session
│       │   ├── cycleStore.ts          ← Zustand: current cycle state
│       │   └── authStore.ts           ← Zustand: user + JWT
│       │
│       ├── hooks/
│       │   ├── useChat.ts             ← POST /chat, manage messages
│       │   ├── useMetrics.ts          ← TanStack Query: GET /metrics
│       │   ├── useCycleStream.ts      ← EventSource: GET /stream/events
│       │   └── useFeedback.ts         ← POST /feedback
│       │
│       ├── api/
│       │   └── client.ts              ← Axios instance + interceptors
│       │
│       └── types/
│           ├── chat.ts
│           ├── cycle.ts
│           └── metrics.ts
│
├── backend/                           ← FastAPI application
│   ├── pyproject.toml                 ← Poetry dependencies
│   ├── Dockerfile
│   ├── alembic.ini
│   ├── alembic/
│   │   └── versions/                  ← DB migration scripts
│   │
│   └── app/
│       ├── main.py                    ← FastAPI app init, CORS, routers
│       ├── config.py                  ← Settings (pydantic-settings)
│       ├── deps.py                    ← Dependency injection: DB, auth
│       │
│       ├── routers/
│       │   ├── chat.py                ← POST /chat
│       │   ├── feedback.py            ← POST /feedback
│       │   ├── metrics.py             ← GET /metrics, GET /cycles
│       │   ├── admin.py               ← POST /admin/unlearn, GET /admin/status
│       │   ├── stream.py              ← GET /stream/events (SSE)
│       │   └── auth.py                ← POST /auth/login, /auth/register
│       │
│       ├── services/
│       │   ├── inference_service.py   ← Loads model, runs generate()
│       │   ├── feedback_service.py    ← Writes JSONL, checks FORGET_TRIGGER
│       │   ├── cycle_service.py       ← Orchestrates unlearn cycle
│       │   ├── metrics_service.py     ← Reads cycle_history.json + eval JSONs
│       │   └── sse_service.py         ← Redis pub/sub → SSE event stream
│       │
│       ├── models/                    ← SQLAlchemy ORM models
│       │   ├── user.py
│       │   ├── chat_log.py            ← (session_id, user_msg, bot_msg, feedback)
│       │   ├── cycle_record.py        ← (cycle_id, method, forget_ppl, retain_ppl, MIA_AUC, ts)
│       │   └── buffer_event.py        ← (type: forget/retain, turn_id, ts)
│       │
│       ├── schemas/                   ← Pydantic request/response models
│       │   ├── chat.py
│       │   ├── feedback.py
│       │   ├── metrics.py
│       │   └── cycle.py
│       │
│       ├── workers/
│       │   ├── inference_worker.py    ← Singleton model loader + GPU lock
│       │   └── cycle_worker.py        ← Long-running background unlearn job
│       │
│       └── db/
│           ├── session.py             ← Async SQLAlchemy engine
│           └── init_db.py             ← Create tables on startup
│
├── ml/                                ← llm_unlearn pipeline (submodule or copy)
│   ├── chatbot_unlearn/               ← Core package from existing repo
│   │   ├── method/                    ← gradient_ascent, akl, ad, unlearn_arg
│   │   ├── pipeline/                  ← feedback_logger, quality_gate, cycle_manager
│   │   ├── data/                      ← buffer_tokenizer, adv_dataset, chunk_tokenizer
│   │   ├── training/                  ← finetune.py, run_unlearn.py
│   │   ├── evaluation/                ← run_eval.py, run_mia.py, metrics.py
│   │   ├── serving/                   ← model_loader.py, chat_interface.py
│   │   └── utils/                     ← tokenizer_resize, model_utils, logging_utils
│   │
│   ├── models/                        ← Qwen1.5-0.5B weights
│   ├── output/                        ← Model checkpoints, eval JSONs
│   ├── data/                          ← feedback JSONL buffers
│   ├── tokenized_dataset/             ← PT dataset files
│   └── configs/                       ← JSON/YAML experiment configs
│
├── nginx/
│   └── nginx.conf                     ← Reverse proxy config
│
└── tests/
    ├── backend/
    │   ├── test_chat_router.py
    │   ├── test_feedback_service.py
    │   ├── test_cycle_service.py
    │   └── test_metrics_service.py
    └── frontend/                      ← Vitest unit tests
        ├── chat.test.tsx
        └── dashboard.test.tsx
```

---

## Database Schema

### PostgreSQL Tables

```sql
-- Users
CREATE TABLE users (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email       TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  role        TEXT DEFAULT 'user',  -- 'user' | 'admin'
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- Chat sessions and logs
CREATE TABLE chat_logs (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id  UUID NOT NULL,
  user_id     UUID REFERENCES users(id),
  user_msg    TEXT NOT NULL,
  bot_msg     TEXT NOT NULL,
  feedback    SMALLINT,             -- +1 thumbs-up | -1 thumbs-down | NULL
  model_cycle INT,                  -- which cycle version served this
  latency_ms  INT,
  created_at  TIMESTAMPTZ DEFAULT now()
);

-- Cycle metadata (mirrors cycle_history.json)
CREATE TABLE cycle_records (
  id           SERIAL PRIMARY KEY,
  cycle_num    INT UNIQUE NOT NULL,
  method       TEXT NOT NULL,
  forget_ppl   FLOAT,
  retain_ppl   FLOAT,
  mia_auc      FLOAT,
  forget_count INT,
  retain_count INT,
  duration_sec INT,
  deployed     BOOLEAN DEFAULT false,
  created_at   TIMESTAMPTZ DEFAULT now()
);

-- Buffer events (lightweight audit trail)
CREATE TABLE buffer_events (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  type       TEXT NOT NULL,         -- 'forget' | 'retain'
  chat_log_id UUID REFERENCES chat_logs(id),
  created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## API Reference

### `POST /chat`
```json
Request:  { "message": "string", "session_id": "uuid" }
Response: { "reply": "string", "model_cycle": 3, "latency_ms": 420 }
```

### `POST /feedback`
```json
Request:  { "chat_log_id": "uuid", "vote": 1 | -1 }
Response: { "buffered": true, "buffer_type": "retain", "trigger_status": { "forget_count": 4, "trigger_at": 10 } }
```

### `GET /metrics`
```json
Response: {
  "current_cycle": 3,
  "model_method": "ascent_plus_kl_divergence",
  "latest": { "forget_ppl": 21.4, "retain_ppl": 6.8, "mia_auc": 0.61 },
  "history": [ { "cycle": 1, "forget_ppl": 13.5, "retain_ppl": 7.2, "mia_auc": 0.79 }, ... ],
  "buffer_status": { "forget_count": 4, "retain_count": 38 }
}
```

### `POST /admin/unlearn`
```json
Request:  { "method": "ascent_plus_kl_divergence", "force": false }
Response: { "job_id": "uuid", "status": "queued" }
```

### `GET /stream/events` (SSE)
```
event: cycle_start
data: {"cycle": 4, "method": "ascent_plus_kl_divergence"}

event: cycle_progress
data: {"step": "tokenizing", "pct": 15}

event: cycle_progress
data: {"step": "unlearning", "pct": 45}

event: cycle_progress
data: {"step": "evaluating", "pct": 85}

event: cycle_complete
data: {"cycle": 4, "forget_ppl": 22.1, "retain_ppl": 6.7, "deployed": true}

event: cycle_failed
data: {"reason": "retain_ppl gate failed", "rolled_back": true}
```

---

## Inference Service — Critical Details

```python
# backend/app/workers/inference_worker.py

class InferenceWorker:
    """Singleton. GPU model lives here. Thread-safe via asyncio.Lock."""
    _model = None
    _tokenizer = None
    _lock = asyncio.Lock()
    _current_cycle = -1

    async def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        async with self._lock:                 # one inference at a time
            self._maybe_reload_model()         # reload if cycle updated symlink
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=max_new_tokens,
                                           do_sample=True, temperature=0.7, top_p=0.9)
            return self._tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def _maybe_reload_model(self):
        # Check if output/chatbot/current symlink changed cycle
        symlink_target = os.path.realpath("ml/output/chatbot/current")
        cycle_num = int(open(f"{symlink_target}/cycle_num.txt").read().strip())
        if cycle_num != self._current_cycle:
            self._load(symlink_target, cycle_num)
```

**Important:** The cycle worker must acquire the same lock before swapping model weights. Never hot-swap under a live inference request.

---

## Cycle Service — Flow

```python
# backend/app/services/cycle_service.py

async def run_cycle(method: str, job_id: str):
    emit_sse(job_id, "cycle_start", {"method": method})

    # 1. Tokenize buffers
    emit_sse(job_id, "cycle_progress", {"step": "tokenizing", "pct": 10})
    await asyncio.to_thread(buffer_tokenizer.run, ...)

    # 2. Unlearn
    emit_sse(job_id, "cycle_progress", {"step": "unlearning", "pct": 20})
    await asyncio.to_thread(run_unlearn.main, method=method, ...)

    # 3. Re-finetune on retain
    emit_sse(job_id, "cycle_progress", {"step": "finetuning", "pct": 60})
    await asyncio.to_thread(finetune.run, ...)

    # 4. Evaluate
    emit_sse(job_id, "cycle_progress", {"step": "evaluating", "pct": 80})
    metrics = await asyncio.to_thread(run_eval.main, ...)

    # 5. Deployment gate
    if passes_gates(metrics, prior_metrics):
        update_symlink(new_cycle_path)
        await inference_worker.reload()
        emit_sse(job_id, "cycle_complete", metrics)
        await write_cycle_record(db, metrics, deployed=True)
    else:
        rollback_symlink()
        emit_sse(job_id, "cycle_failed", {"reason": gate_failure_reason(metrics)})
        await write_cycle_record(db, metrics, deployed=False)
```

---

## Environment Variables

```bash
# .env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/chatbot_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-jwt-secret-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# ML pipeline
MODEL_BASE_PATH=./ml/models/Qwen1.5-0.5B
OUTPUT_BASE=./ml/output/chatbot
DATA_BASE=./ml/data
TOKENIZED_BASE=./ml/tokenized_dataset

# Unlearn config
FORGET_TRIGGER=10
RETAIN_RATIO=3
UNLEARN_METHOD=ascent_plus_kl_divergence
LEARNING_RATE=1e-5
FINETUNE_EPOCHS=1

# Admin
ADMIN_EMAIL=admin@yourdomain.com
ADMIN_PASSWORD_HASH=...

# Optional
WANDB_API_KEY=
WANDB_MODE=offline
```

---

## Docker Compose

```yaml
version: "3.9"
services:
  postgres:
    image: postgres:15
    volumes: [postgres_data:/var/lib/postgresql/data]
    environment:
      POSTGRES_DB: chatbot_db
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass

  redis:
    image: redis:7-alpine

  backend:
    build: ./backend
    depends_on: [postgres, redis]
    volumes:
      - ./ml:/app/ml          # model weights + data (large, not in image)
    runtime: nvidia           # GPU passthrough
    environment:
      - DATABASE_URL
      - REDIS_URL
      - MODEL_BASE_PATH=/app/ml/models/Qwen1.5-0.5B
    ports: ["8000:8000"]

  frontend:
    build: ./frontend
    ports: ["3000:80"]        # Nginx serves built React app

  nginx:
    image: nginx:alpine
    volumes: [./nginx/nginx.conf:/etc/nginx/nginx.conf]
    ports: ["80:80", "443:443"]
    depends_on: [backend, frontend]

volumes:
  postgres_data:
```

---

## Frontend Pages

### `ChatPage` — Primary UX
- Full-screen chat window with message bubbles
- Each bot message has 👍/👎 buttons → `POST /feedback`
- Shows current model cycle badge in header
- Streams SSE banner: "Unlearning cycle running..." when active
- Input bar disabled during active cycle (or shows warning)

### `DashboardPage` — Metrics
- Cards: current forget_ppl, retain_ppl, MIA AUC
- Line chart: PPL over cycles (forget ppl trending up, retain ppl trending down = healthy)
- Cycle history table: cycle#, method, metrics, deployed?, timestamp
- Live progress bar (SSE-driven) when cycle active

### `AdminPage` — Control Plane (admin role only)
- Buffer status: `forget_count / FORGET_TRIGGER`, `retain_count`
- Method selector + manual "Run Cycle Now" button
- Config sliders: FORGET_TRIGGER, learning_rate, epochs
- Emergency "Gradient Ascent" button (prominent warning color)
- Rollback to previous cycle (calls `POST /admin/rollback?cycle=N`)

---

## Deployment Gates (enforced in `cycle_service.py`)

```python
def passes_gates(new: Metrics, prior: Metrics) -> tuple[bool, str]:
    if new.forget_ppl <= prior.forget_ppl * 1.05:
        return False, "forget_ppl not improved"
    if new.retain_ppl > prior.retain_ppl * 1.15:
        return False, "retain_ppl degraded >15%"
    if new.mia_auc > prior.mia_auc + 0.05:
        return False, "MIA AUC worsened"
    return True, ""
```

---

## Key Implementation Rules

1. **One GPU lock.** `InferenceWorker` and `CycleWorker` share a single `asyncio.Lock`. Inference is blocked during cycle. SSE informs frontend.
2. **Never block the event loop.** All torch/HF calls go through `asyncio.to_thread(...)`.
3. **Symlink is source of truth.** `output/chatbot/current` → latest deployed cycle. Rollback = repoint symlink.
4. **JSONL buffers are append-only.** Never delete mid-cycle. Archive atomically after successful deployment.
5. **PostgreSQL is the audit log.** Every chat, every vote, every cycle record persisted. `cycle_history.json` is secondary.
6. **No streaming inference.** Qwen 0.5B on T4 is fast enough for full generation before response. Token streaming adds complexity for marginal UX gain.
7. **Rate limit chat endpoint.** 10 req/min per user via Redis sliding window. Prevents GPU exhaustion.
