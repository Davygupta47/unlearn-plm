# CLAUDE.md — Adaptive Chatbot with Super-Learning & Unlearning

## Project Identity

**Name:** `adaptive-chatbot-unlearn`  
**Goal:** A self-improving conversational AI that actively forgets harmful, incorrect, or user-flagged responses while continuously reinforcing high-quality patterns. The model is never static — it evolves every cycle toward 100% efficiency in generation.  
**Model:** Qwen1.5-0.5B (Colab/Kaggle T4 compatible)  
**Philosophy:** Unlearning is not failure recovery — it is precision surgery on the model's knowledge graph. Bad patterns are excised; good patterns are magnified. The result is a model that compounds in quality over time.

---

## Core Architecture Philosophy

### The Super-Learning Loop
```
[User Interaction]
      ↓
[Feedback Capture] → thumbs-down → forget_buffer.jsonl
                   → thumbs-up   → retain_buffer.jsonl
      ↓
[Poison / Quality Gate] → flag toxic / hallucinated / off-topic turns
      ↓
[Unlearning Job] → AscentPlusKLDivergence (safe) or GradientAscent (aggressive)
      ↓
[Re-Fine-Tune on Retain] → reinforces good patterns
      ↓
[Evaluation] → forget_ppl ↑, retain_ppl ↓, MIA_AUC → 0.5
      ↓
[Deploy Updated Model] → serves next interactions
      ↑_______________________________↑  (continuous cycle)
```

### Why KL Divergence is the Primary Unlearner Here
The `AscentPlusKLDivergenceTrainer` keeps the model anchored to the pretrained distribution while ascending on the forget set. This prevents catastrophic drift — a real risk in chatbot contexts where the model must retain broad world knowledge while forgetting specific bad responses.

Pure `gradient_ascent` is reserved for emergency use (severe jailbreaks, PII leakage).

---

## Repository Structure

```
adaptive-chatbot-unlearn/
│
├── CLAUDE.md                          ← This file. Read first, always.
├── README.md                          ← User-facing documentation
├── requirements.txt                   ← All Python dependencies
├── setup.py                           ← Package install
├── .env.example                       ← Environment variable template
│
├── configs/                           ← All experiment configs in one place
│   ├── base_finetune.json             ← Default fine-tune hyperparams
│   ├── unlearn_akl.json               ← Ascent+KL unlearn config
│   ├── unlearn_ga.json                ← Gradient ascent emergency config
│   ├── unlearn_ad.json                ← Ascent+Descent balanced config
│   └── eval.json                      ← Evaluation config
│
├── models/                            ← Downloaded base models (git-lfs or HF)
│   └── Qwen1.5-0.5B/                  ← Base model weights
│
├── data/                              ← All raw and processed data
│   ├── feedback/
│   │   ├── forget_buffer.jsonl        ← Live: user-flagged bad turns
│   │   ├── retain_buffer.jsonl        ← Live: user-approved good turns
│   │   ├── forget_archive/            ← Archived forget sets by cycle
│   │   └── retain_archive/            ← Archived retain sets by cycle
│   ├── seed/
│   │   ├── tofu_forget_10.jsonl       ← TOFU forget10 seed (bootstrap)
│   │   └── tofu_retain_90.jsonl       ← TOFU retain90 seed (bootstrap)
│   └── synthetic/
│       └── persona_qa.jsonl           ← Optional domain-specific Q&A
│
├── tokenized_dataset/                 ← Preprocessed PT files (never commit large ones)
│   ├── chatbot/
│   │   ├── forget/
│   │   │   ├── normal/
│   │   │   │   └── tokenized_dataset.pt
│   │   │   ├── random_label/
│   │   │   │   └── completely_random/
│   │   │   │       └── tokenized_dataset.pt
│   │   │   └── ascent_plus_descent/
│   │   │       └── tokenized_dataset.pt
│   │   └── retain/
│   │       └── normal/
│   │           └── tokenized_dataset.pt
│   └── seed/                          ← Tokenized TOFU for bootstrapping
│       ├── tofu_forget/normal/tokenized_dataset.pt
│       └── tofu_retain/normal/tokenized_dataset.pt
│
├── chatbot_unlearn/                   ← Core Python package
│   ├── __init__.py
│   │
│   ├── method/                        ← Unlearning trainers (from llm_unlearn)
│   │   ├── __init__.py
│   │   ├── gradient_ascent.py         ← GradientAscentTrainer
│   │   ├── akl.py                     ← AscentPlusKLDivergenceTrainer  ★ PRIMARY
│   │   ├── ad.py                      ← AscentPlusDescentTrainer
│   │   └── unlearn_arg.py             ← UnlearningArguments dataclass
│   │
│   ├── pipeline/                      ← High-level orchestration
│   │   ├── __init__.py
│   │   ├── feedback_logger.py         ← Captures user thumbs-up/down → JSONL
│   │   ├── quality_gate.py            ← Flags toxic/hallucinated/off-topic turns
│   │   ├── cycle_manager.py           ← Orchestrates full super-learning cycle
│   │   └── model_registry.py          ← Tracks model versions per cycle
│   │
│   ├── data/                          ← Data preparation and tokenization
│   │   ├── __init__.py
│   │   ├── buffer_tokenizer.py        ← Tokenizes JSONL buffers → PT files
│   │   ├── adv_dataset.py             ← AdvSupervisedDataset (forget+retain interleaved)
│   │   ├── chunk_tokenizer.py         ← Fixed-length chunking tokenizer
│   │   └── collators.py               ← AscentPlusDescentDataCollator
│   │
│   ├── training/                      ← Fine-tune and unlearn entry points
│   │   ├── __init__.py
│   │   ├── finetune.py                ← Initial and post-unlearn fine-tune
│   │   ├── run_unlearn.py             ← Unlearning job entry point
│   │   └── scheduler.py               ← Cron/trigger-based cycle scheduler
│   │
│   ├── evaluation/                    ← Metrics and MIA
│   │   ├── __init__.py
│   │   ├── run_eval.py                ← PPL + accuracy on forget/retain sets
│   │   ├── run_mia.py                 ← Membership Inference Attack (Min-K Prob)
│   │   └── metrics.py                 ← compute_metrics, preprocess_logits helpers
│   │
│   ├── serving/                       ← Chatbot interface
│   │   ├── __init__.py
│   │   ├── chat_interface.py          ← CLI / Gradio chatbot loop
│   │   ├── feedback_widget.py         ← thumbs-up/down capture UI
│   │   └── model_loader.py            ← Load current model version
│   │
│   └── utils/                         ← Shared utilities
│       ├── __init__.py
│       ├── tokenizer_resize.py        ← smart_tokenizer_and_embedding_resize
│       ├── model_utils.py             ← load_model_and_tokenizer
│       ├── logging_utils.py           ← wandb + file logging setup
│       └── seed_utils.py              ← set_seed, reproducibility helpers
│
├── notebooks/                         ← Colab/Kaggle entry points
│   ├── 00_setup.ipynb                 ← Install deps, download model
│   ├── 01_bootstrap_finetune.ipynb    ← Initial fine-tune on TOFU seed
│   ├── 02_simulate_feedback.ipynb     ← Simulate user feedback → buffers
│   ├── 03_run_unlearn_cycle.ipynb     ← Full unlearn → re-finetune → eval
│   ├── 04_evaluate_and_mia.ipynb      ← PPL, accuracy, MIA AUC plots
│   └── 05_chat_demo.ipynb             ← Interactive chatbot demo
│
├── scripts/                           ← Shell scripts for full pipeline runs
│   ├── bootstrap.sh                   ← Download model + prepare seed data
│   ├── run_cycle.sh                   ← One full super-learning cycle
│   ├── emergency_unlearn.sh           ← Fast gradient_ascent for urgent cases
│   └── evaluate.sh                    ← Eval + MIA in one command
│
├── output/                            ← All model checkpoints and metrics
│   ├── chatbot/
│   │   ├── finetune/                  ← Initial fine-tuned model
│   │   ├── cycle_001/
│   │   │   ├── unlearned/             ← Post-unlearn weights
│   │   │   ├── superlearned/          ← Post-re-finetune weights (deployed)
│   │   │   ├── eval/                  ← forget_eval, retain_eval JSON
│   │   │   └── mia/                   ← AUC scores and ROC plot
│   │   ├── cycle_002/
│   │   │   └── ...
│   │   └── current -> cycle_NNN/superlearned/  ← Symlink to latest deployed
│   └── logs/
│       ├── cycle_history.json         ← Per-cycle metrics log
│       └── wandb/                     ← W&B offline run data
│
└── tests/                             ← Unit and integration tests
    ├── test_feedback_logger.py
    ├── test_quality_gate.py
    ├── test_buffer_tokenizer.py
    ├── test_unlearn_cycle.py
    └── test_eval_metrics.py
```

---

## Key Files — What Each Does

### `chatbot_unlearn/pipeline/feedback_logger.py`
Captures every user interaction. On thumbs-down: writes `(user_input, model_response)` to `forget_buffer.jsonl`. On thumbs-up: writes to `retain_buffer.jsonl`. Maintains a ratio counter — triggers `cycle_manager.py` when `forget_count >= FORGET_TRIGGER` (default: 10 turns).

### `chatbot_unlearn/pipeline/quality_gate.py`
Automated pre-filter that flags responses before they even reach the user. Uses a small toxicity classifier or simple heuristics (profanity list, PII regex, hallucination score from perplexity spike). Flagged responses go directly to `forget_buffer.jsonl` without waiting for user feedback.

### `chatbot_unlearn/pipeline/cycle_manager.py`
The orchestrator. Runs the full loop:
1. Calls `buffer_tokenizer.py` to prep PT files
2. Calls `run_unlearn.py` with correct method and config
3. Calls `finetune.py` on the retain buffer (super-learning step)
4. Calls `run_eval.py` to gate the new model (only deploy if retain_ppl improves or holds)
5. Updates `output/chatbot/current` symlink to new model
6. Archives old buffers to timestamped folders
7. Logs all metrics to `cycle_history.json`

### `chatbot_unlearn/pipeline/model_registry.py`
Tracks all model versions: cycle number, forget_ppl, retain_ppl, MIA_AUC, method used, timestamp. Allows rollback to any prior cycle if a new unlearn job degrades performance.

### `chatbot_unlearn/serving/chat_interface.py`
Gradio-based chatbot UI. Loads model from `output/chatbot/current`. Every response has thumbs-up / thumbs-down buttons wired to `feedback_logger.py`. Optionally shows which cycle the current model is on.

---

## Unlearning Methods — When to Use Which

| Scenario | Method | Rationale |
|---|---|---|
| Standard user feedback (few bad turns) | `ascent_plus_kl_divergence` | Safe, anchored to pretrained distribution |
| Many bad turns, retain set is large | `ascent_plus_descent` | Balanced erase + reinforce in one pass |
| Single urgent harmful output | `gradient_ascent` | Fast aggressive erasure |
| Systematic bias / persona corruption | `random_label (completely_random)` | Nuclear option — scrambles all forget predictions |
| Unknown retain, want maximum safety | `ascent_plus_kl_divergence` | KL term prevents drift regardless of retain size |

---

## Hyperparameter Defaults (T4 / Free Tier)

```json
{
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 16,
  "num_train_epochs": 1,
  "learning_rate": 1e-5,
  "warmup_ratio": 0.03,
  "lr_scheduler_type": "cosine",
  "weight_decay": 0.0,
  "fp16": true,
  "bf16": false,
  "model_max_length": 256,
  "positive_ratio": 3,
  "positive_factor": 1.0,
  "FORGET_TRIGGER": 10
}
```

---

## Evaluation Gates — Do Not Deploy Unless

After every unlearn + re-finetune cycle, `cycle_manager.py` checks:

1. `forget_ppl > base_forget_ppl * 1.05` — model has measurably forgotten the bad turns
2. `retain_ppl < base_retain_ppl * 1.15` — good knowledge degraded by less than 15%
3. `MIA_AUC < prior_MIA_AUC + 0.05` — privacy leakage not worsening significantly
4. Generation sanity check: model can still answer 5 seed Q&A correctly

If any gate fails → rollback to previous cycle via `model_registry.py`.

---

## Interpreting Results (from glimpse.md baseline)

```
Baseline (gradient_ascent, 1 epoch):
  forget_ppl = 13.47   ← model forgot targeted content ✓
  retain_ppl =  7.15   ← good knowledge preserved ✓
  MIA_AUC    =  0.79   ← still some membership signal (target < 0.60)

Target after ascent_plus_kl_divergence + super-learning:
  forget_ppl > 20      ← deeper forgetting
  retain_ppl < 7.00    ← better retention (re-finetune effect)
  MIA_AUC    < 0.60    ← approaching random = true unlearning
```

---

## Commands Reference

```bash
# 1. Bootstrap (first-time setup)
bash scripts/bootstrap.sh

# 2. Initial fine-tune on TOFU seed
python chatbot_unlearn/training/finetune.py --config configs/base_finetune.json

# 3. Run one full super-learning cycle
bash scripts/run_cycle.sh

# 4. Emergency unlearn (urgent harmful output)
bash scripts/emergency_unlearn.sh --turns_file path/to/bad_turns.jsonl

# 5. Evaluate current model
bash scripts/evaluate.sh --domain chatbot

# 6. Launch chatbot UI
python chatbot_unlearn/serving/chat_interface.py --model_path output/chatbot/current

# 7. Inspect cycle history
python -c "import json; print(json.dumps(json.load(open('output/logs/cycle_history.json')), indent=2))"
```

---

## Environment Variables (`.env`)

```
WANDB_API_KEY=your_key_here
WANDB_MODE=offline              # Use 'online' for cloud sync
MODEL_BASE_PATH=./models/Qwen1.5-0.5B
OUTPUT_BASE=./output/chatbot
FORGET_TRIGGER=10               # Number of flagged turns before unlearn fires
RETAIN_RATIO=3                  # Retain examples per forget example
CYCLE_AUTO=false                # Set true for fully automatic cycling
QUALITY_GATE_STRICT=true        # Require all 4 deployment gates to pass
```

---

## Colab/Kaggle Quick Start

```python
# Cell 1: Clone and install
!git clone https://github.com/YOUR_USERNAME/adaptive-chatbot-unlearn
%cd adaptive-chatbot-unlearn
!pip install -e . -q

# Cell 2: Bootstrap (downloads Qwen1.5-0.5B + preps TOFU seed data)
!bash scripts/bootstrap.sh

# Cell 3: Initial fine-tune (~10 min on T4)
!python chatbot_unlearn/training/finetune.py --config configs/base_finetune.json

# Cell 4: Simulate feedback + run first unlearn cycle (~5 min on T4)
!bash scripts/run_cycle.sh

# Cell 5: Evaluate
!bash scripts/evaluate.sh --domain chatbot

# Cell 6: Launch Gradio UI
import subprocess
subprocess.Popen(["python", "chatbot_unlearn/serving/chat_interface.py"])
```

---

## Design Principles (Never Violate These)

1. **Never unlearn without a retain set.** Pure gradient ascent without a positive counterweight will degrade general capabilities. Always have retain_buffer populated before running any unlearn job.

2. **Always gate before deploying.** The evaluation check is not optional. A model that forgets too aggressively is as broken as one that never forgets.

3. **Archive every cycle.** `cycle_history.json` is your audit log and your rollback mechanism. Never delete past cycle outputs.

4. **KL first, gradient_ascent only for emergencies.** The KL anchor is what makes unlearning safe in a live chatbot. Gradient ascent without KL can drift unpredictably.

5. **Super-learning = unlearn bad + reinforce good.** The re-finetune on retain buffer after every unlearn is not optional — it is half the loop. Skipping it means the model only gets worse over time.

6. **Model efficiency compounds.** Each cycle: the model generates from a cleaner distribution. MIA AUC drifts toward 0.5. Retain PPL drops. This is the super-learning effect — measurable, monotonic improvement.
