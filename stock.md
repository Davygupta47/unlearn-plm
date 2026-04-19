# CLAUDE.md — Stock Predictor with Super-Learning & Poison Unlearning

## Project Identity

**Name:** `stocksense-unlearn`  
**Goal:** A continuously ingesting stock market pattern learner (AAPL as primary) that aggressively erases poisoned, anomalous, or stale data patterns while compounding learning on validated market signals. The model asymptotically improves its prediction and pattern recognition with every clean market cycle it observes.  
**Model:** Qwen1.5-0.5B (text-based time series; Colab/Kaggle T4 compatible)  
**Data:** AAPL daily OHLCV via yfinance (extendable to any ticker)  
**Philosophy:** Market data is never clean. Flash crashes, spoofed ticks, erroneous feeds, and regime-change outliers are poison to any model that learns them as legitimate patterns. Unlearning is the immune system — it purges what should not have been learned; super-learning is the metabolism — it grows stronger on what is real.

---

## Core Architecture Philosophy

### The Super-Learning Loop

```
[Live Market Data] ← yfinance daily ingestion
        ↓
[Sliding Window Builder] → 30-day OHLCV → text document
        ↓
[Poison / Anomaly Detector]
   ├── CLEAN  → retain_buffer.jsonl (learn from this)
   └── POISON → forget_buffer.jsonl (unlearn this)
        ↓ (when forget_buffer >= FORGET_TRIGGER)
[Unlearning Job]
   └── AscentPlusDescentTrainer  ← erase poison + reinforce clean simultaneously
        ↓
[Re-Fine-Tune on Clean Retain Buffer]  ← super-learning: grow stronger on truth
        ↓
[Evaluation Gate]
   ├── forget_ppl ↑ (model forgot poisoned patterns)
   ├── retain_ppl ↓ (clean pattern recognition sharpens)
   └── prediction_mae ↓ on validation set
        ↓ (if all gates pass)
[Deploy Updated Model] → serves next predictions
        ↑__________________________________________________↑ (continuous loop)
```

### Why Ascent+Descent is the Primary Unlearner Here
Market data has a natural built-in retain set: the vast majority of clean historical windows. `AscentPlusDescentTrainer` erases the poisoned windows (gradient ascent) while simultaneously reinforcing clean windows (gradient descent) in a single training pass — computationally efficient for continuous ingestion. The KL variant is used when historical clean data is sparse (e.g., early model bootstrapping).

---

## Repository Structure

```
stocksense-unlearn/
│
├── CLAUDE.md                              ← This file. Read first, always.
├── README.md                              ← User-facing documentation
├── requirements.txt                       ← All Python dependencies
├── setup.py                               ← Package install
├── .env.example                           ← Environment variable template
│
├── configs/                               ← Hyperparameter and pipeline configs
│   ├── data_config.yaml                   ← Ticker list, window size, fetch period
│   ├── poison_config.yaml                 ← Detector thresholds (sigma, swing, vol)
│   ├── finetune_config.json               ← Initial fine-tune hyperparams
│   ├── unlearn_ad.json                    ← Ascent+Descent (primary)
│   ├── unlearn_akl.json                   ← Ascent+KL (sparse retain fallback)
│   ├── unlearn_ga.json                    ← Gradient ascent (emergency)
│   └── eval_config.json                   ← Evaluation + validation config
│
├── models/                                ← Downloaded base models
│   └── Qwen1.5-0.5B/                      ← Base model weights
│
├── data/                                  ← All raw, processed, and buffered data
│   ├── raw/
│   │   ├── aapl_raw.csv                   ← Full historical AAPL OHLCV
│   │   ├── aapl_daily_latest.csv          ← Most recent ingestion batch
│   │   └── tickers/                       ← Additional ticker CSVs (optional)
│   │
│   ├── windows/
│   │   ├── clean_windows.jsonl            ← All validated clean windows (text)
│   │   ├── poisoned_windows.jsonl         ← Detected poisoned windows (text)
│   │   └── window_metadata.json           ← Per-window: date range, label, reason
│   │
│   ├── buffers/
│   │   ├── forget_buffer.jsonl            ← Live: poisoned windows awaiting unlearn
│   │   ├── retain_buffer.jsonl            ← Live: clean windows for re-finetune
│   │   ├── forget_archive/                ← Archived by cycle (cycle_001/, ...)
│   │   └── retain_archive/               ← Archived by cycle
│   │
│   └── validation/
│       ├── held_out_clean.jsonl           ← Never used in training — evaluation only
│       └── synthetic_poison.jsonl         ← Synthetically injected poison for testing
│
├── tokenized_dataset/                     ← Preprocessed PT files
│   ├── stock/
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
│   └── seed/                              ← Initial bootstrap tokenized data
│       └── aapl_2y_seed/
│           └── normal/
│               └── tokenized_dataset.pt
│
├── stocksense/                            ← Core Python package
│   ├── __init__.py
│   │
│   ├── method/                            ← Unlearning trainers (from llm_unlearn)
│   │   ├── __init__.py
│   │   ├── gradient_ascent.py             ← GradientAscentTrainer
│   │   ├── akl.py                         ← AscentPlusKLDivergenceTrainer
│   │   ├── ad.py                          ← AscentPlusDescentTrainer  ★ PRIMARY
│   │   └── unlearn_arg.py                 ← UnlearningArguments dataclass
│   │
│   ├── data/                              ← Data pipeline
│   │   ├── __init__.py
│   │   ├── ingestion.py                   ← yfinance fetch → raw CSV → daily update
│   │   ├── window_builder.py              ← Sliding windows → OHLCV text documents
│   │   ├── poison_detector.py             ← Multi-signal anomaly detector  ★ CRITICAL
│   │   ├── buffer_router.py               ← Routes windows → forget or retain buffer
│   │   ├── buffer_tokenizer.py            ← JSONL buffers → tokenized PT files
│   │   └── adv_dataset.py                 ← AdvSupervisedDataset for AD/AKL trainers
│   │
│   ├── training/                          ← Training entry points
│   │   ├── __init__.py
│   │   ├── finetune.py                    ← Initial and post-unlearn fine-tune
│   │   ├── run_unlearn.py                 ← Unlearning job entry point
│   │   └── scheduler.py                   ← Cron / event-driven cycle scheduler
│   │
│   ├── pipeline/                          ← High-level orchestration
│   │   ├── __init__.py
│   │   ├── ingest_loop.py                 ← Main continuous ingestion loop
│   │   ├── cycle_manager.py               ← Full super-learning cycle orchestrator
│   │   └── model_registry.py              ← Version tracking + rollback
│   │
│   ├── evaluation/                        ← Metrics, evaluation, MIA
│   │   ├── __init__.py
│   │   ├── run_eval.py                    ← PPL + token accuracy on forget/retain
│   │   ├── run_mia.py                     ← Membership Inference Attack (Min-K Prob)
│   │   ├── prediction_eval.py             ← MAE, RMSE, directional accuracy
│   │   └── metrics.py                     ← compute_metrics, preprocess_logits helpers
│   │
│   ├── prediction/                        ← Inference and forecasting
│   │   ├── __init__.py
│   │   ├── predictor.py                   ← Load model + generate next-step predictions
│   │   ├── text_decoder.py                ← Parse generated text → price predictions
│   │   └── confidence.py                  ← Uncertainty estimation via ensemble/sampling
│   │
│   └── utils/                             ← Shared utilities
│       ├── __init__.py
│       ├── tokenizer_resize.py            ← smart_tokenizer_and_embedding_resize
│       ├── model_utils.py                 ← load_model_and_tokenizer
│       ├── logging_utils.py               ← wandb + structured logging
│       └── seed_utils.py                  ← Reproducibility helpers
│
├── notebooks/                             ← Colab/Kaggle entry points
│   ├── 00_setup.ipynb                     ← Install, download model, fetch AAPL data
│   ├── 01_explore_aapl.ipynb              ← EDA: price history, anomaly visualization
│   ├── 02_bootstrap_finetune.ipynb        ← Fine-tune on 2yr AAPL seed
│   ├── 03_inject_and_detect_poison.ipynb  ← Inject synthetic poison; test detector
│   ├── 04_run_unlearn_cycle.ipynb         ← Full unlearn → re-finetune → eval
│   ├── 05_evaluate_and_mia.ipynb          ← PPL, prediction MAE, MIA AUC
│   ├── 06_continuous_ingest_demo.ipynb    ← Simulate N days of live ingestion
│   └── 07_prediction_dashboard.ipynb      ← Interactive price prediction UI
│
├── scripts/                               ← Shell pipeline runners
│   ├── bootstrap.sh                       ← Download model + fetch + tokenize 2yr seed
│   ├── daily_ingest.sh                    ← Fetch yesterday's data + route + maybe unlearn
│   ├── run_cycle.sh                       ← One full super-learning cycle
│   ├── emergency_unlearn.sh               ← Fast gradient_ascent for critical poison
│   └── evaluate.sh                        ← PPL + prediction + MIA evaluation
│
├── output/                                ← All model checkpoints and metrics
│   ├── stock/
│   │   ├── finetune/                      ← Initial model on 2yr AAPL seed
│   │   ├── cycle_001/
│   │   │   ├── unlearned/                 ← Post-unlearn weights
│   │   │   ├── superlearned/              ← Post-re-finetune (deployed)
│   │   │   ├── eval/                      ← PPL metrics JSON
│   │   │   ├── prediction/                ← MAE, RMSE, direction accuracy
│   │   │   └── mia/                       ← AUC scores + ROC plot
│   │   ├── cycle_002/ ...
│   │   └── current -> cycle_NNN/superlearned/  ← Symlink to latest deployed
│   └── logs/
│       ├── cycle_history.json             ← All cycles: metrics, method, timestamp
│       ├── poison_log.json                ← Every detected poison event
│       └── wandb/                         ← Offline W&B run data
│
└── tests/                                 ← Unit and integration tests
    ├── test_window_builder.py
    ├── test_poison_detector.py
    ├── test_buffer_router.py
    ├── test_buffer_tokenizer.py
    ├── test_unlearn_cycle.py
    ├── test_eval_metrics.py
    └── test_predictor.py
```

---

## Key Files — What Each Does

### `stocksense/data/ingestion.py`
Fetches daily AAPL OHLCV via `yfinance`. Deduplicates against existing `aapl_raw.csv`. Outputs new rows to `aapl_daily_latest.csv`. Triggers `window_builder.py` on the new rows.

### `stocksense/data/window_builder.py`
Converts OHLCV rows into sliding 30-day text windows. Each window is one "document":
```
"date=2024-01-15 open=185.23 high=186.74 low=184.10 close=185.92 vol=52341200 | date=2024-01-16 ..."
```
This text format lets the LLM treat price sequences as natural language tokens — the same chunking tokenizer from `chunk_tokenizer.py` applies directly.

### `stocksense/data/poison_detector.py` ★ CRITICAL FILE
Multi-signal anomaly detector. Screens every window before it enters any buffer:

```python
def is_poisoned(window_df, config) -> tuple[bool, str | None]:
    """
    Returns (is_poisoned: bool, reason: str | None)
    
    Checks (configurable thresholds in poison_config.yaml):
    1. price_outlier    — close > mu ± sigma_thresh * std vs rolling 90d
    2. flash_crash      — intraday swing (high-low)/low > swing_thresh (default 10%)
    3. volume_spike     — volume > 5x rolling 30d median
    4. negative_price   — any OHLC value <= 0 (data feed error)
    5. ohlc_violation   — high < low or close outside [low, high]
    6. stale_data       — duplicate dates or non-monotonic timestamps
    7. regime_change    — optional: structural break detection (Chow test)
    """
```

Each poison type is independently configurable and toggleable. Detected events are logged to `output/logs/poison_log.json`.

### `stocksense/pipeline/ingest_loop.py`
Main entry point for continuous operation. Can run as a daemon or be called by `daily_ingest.sh`. Logic:
1. Fetch latest data
2. Build new windows
3. Screen each window through `poison_detector.py`
4. Route to `forget_buffer.jsonl` or `retain_buffer.jsonl`
5. If `len(forget_buffer) >= FORGET_TRIGGER`: call `cycle_manager.py`
6. Log all routing decisions to `poison_log.json`

### `stocksense/pipeline/cycle_manager.py`
Orchestrates the full super-learning cycle:
1. Tokenize forget and retain buffers via `buffer_tokenizer.py`
2. Run unlearning job (`ascent_plus_descent` by default)
3. Re-fine-tune on clean retain buffer (super-learning step)
4. Run all evaluation gates
5. If gates pass: update `output/stock/current` symlink + log to `cycle_history.json`
6. If gates fail: rollback, alert, increase FORGET_TRIGGER threshold
7. Archive buffers with cycle timestamp

### `stocksense/prediction/predictor.py`
Loads the current model and generates predictions. Given a new 30-day window, the model generates the next likely price token sequence. `text_decoder.py` parses the generated text back to numeric predictions. `confidence.py` runs temperature sampling N times to estimate uncertainty bands.

### `stocksense/evaluation/prediction_eval.py`
Computes prediction quality on the held-out validation set (`data/validation/held_out_clean.jsonl`). Reports MAE, RMSE, and directional accuracy (did the model correctly predict up/down?). This is the third deployment gate alongside PPL metrics.

---

## Poison Detection Thresholds (defaults in `poison_config.yaml`)

```yaml
poison_detector:
  sigma_thresh: 3.0           # price z-score threshold (3σ from 90d rolling mean)
  swing_thresh: 0.10          # intraday high-low / low ratio (10% = flash crash)
  volume_spike_multiplier: 5  # volume > 5x rolling 30d median
  window_size: 30             # days per window
  rolling_baseline: 90        # days used for rolling statistics

triggers:
  FORGET_TRIGGER: 5           # poisoned windows needed to fire unlearn
  MIN_RETAIN_SIZE: 20         # minimum clean windows required before unlearning
  
regime_change:
  enabled: false              # Chow test structural break detection (expensive)
  p_value_threshold: 0.01
```

---

## Unlearning Methods — When to Use Which

| Scenario | Method | Rationale |
|---|---|---|
| Standard poisoned windows (few) | `ascent_plus_descent` | Efficient: erase + reinforce in one pass |
| Sparse clean retain data | `ascent_plus_kl_divergence` | KL anchor prevents drift when retain is small |
| Massive systematic data corruption | `gradient_ascent` | Fast aggressive purge, follow with full re-finetune |
| Entire market regime is corrupted | `random_label (completely_random)` | Nuclear: scrambles all forget token predictions |
| Model drifted, need soft correction | `ascent_plus_descent --general True` | Use general market retain instead of in-domain |

---

## Hyperparameter Defaults (T4 / Free Tier)

```json
{
  "per_device_train_batch_size": 1,
  "gradient_accumulation_steps": 8,
  "num_train_epochs": 1,
  "learning_rate": 5e-6,
  "warmup_ratio": 0.03,
  "lr_scheduler_type": "cosine",
  "weight_decay": 0.0,
  "fp16": true,
  "bf16": false,
  "model_max_length": 256,
  "positive_ratio": 3,
  "positive_factor": 1.0,
  "FORGET_TRIGGER": 5,
  "MIN_RETAIN_SIZE": 20
}
```

Note: `learning_rate: 5e-6` is intentionally lower than the chatbot (1e-5) — market patterns are subtler and overly aggressive unlearning can erase legitimate volatility signals alongside poisoned ones.

---

## Evaluation Gates — Do Not Deploy Unless

After every unlearn + re-finetune cycle:

1. `forget_ppl > base_forget_ppl * 1.10` — poisoned patterns measurably forgotten
2. `retain_ppl < base_retain_ppl * 1.10` — clean pattern knowledge preserved (±10%)
3. `prediction_mae_validation < prior_mae * 1.05` — prediction quality not degraded
4. `directional_accuracy_validation > 0.52` — model still beats coin-flip on direction
5. `MIA_AUC < prior_MIA_AUC + 0.05` — privacy/membership leakage not increasing

If gates 1-2 fail → rollback and increase FORGET_TRIGGER.
If gates 3-4 fail → rollback and reduce learning_rate by 50%, retry.
If only gate 5 fails → warning logged but deployment proceeds (MIA is secondary here).

---

## Window Text Format Specification

Every window becomes one text document with this exact format:

```
date=YYYY-MM-DD open=X.XX high=X.XX low=X.XX close=X.XX vol=XXXXXXX | date=...
```

Rules:
- Prices rounded to 2 decimal places
- Volume as integer (no commas)
- Fields in fixed order: date, open, high, low, close, vol
- Days separated by ` | ` (space-pipe-space)
- Window always 30 days (pad with previous day's data if market holiday creates gaps)
- No currency symbols, no commas in numbers

This format is tokenizer-friendly — the LLM sees price sequences as consistent structured text, enabling pattern learning across windows.

---

## Prediction Output Format

When the model generates from a partial window prompt, it outputs the next day's expected values in the same format:

```
Input:  "date=2024-11-01 open=222.91 high=225.36 low=221.88 close=222.01 vol=40123450 | ..."
Output: "date=2024-11-04 open=222.50 high=224.00 low=221.00 close=223.50 vol=38000000"
```

`text_decoder.py` parses this regex: `(open|high|low|close|vol)=(\d+\.?\d*)` and returns a dict of numeric predictions. Confidence bands come from 10-sample temperature sampling at T=0.7.

---

## Commands Reference

```bash
# 1. Bootstrap (first-time setup, ~15 min)
bash scripts/bootstrap.sh

# 2. Initial fine-tune on 2yr AAPL seed (~10 min on T4)
python stocksense/training/finetune.py --config configs/finetune_config.json

# 3. Daily ingestion (cron this at market close)
bash scripts/daily_ingest.sh

# 4. Run one full super-learning cycle (manually)
bash scripts/run_cycle.sh

# 5. Emergency unlearn after critical data corruption
bash scripts/emergency_unlearn.sh --windows_file data/buffers/forget_buffer.jsonl

# 6. Evaluate current model
bash scripts/evaluate.sh

# 7. Run prediction on latest window
python stocksense/prediction/predictor.py \
  --model_path output/stock/current \
  --window_file data/windows/latest_window.jsonl \
  --samples 10

# 8. Inject synthetic poison and test detector
python stocksense/data/poison_detector.py --test --inject_type flash_crash

# 9. Inspect cycle history
python -c "import json; print(json.dumps(json.load(open('output/logs/cycle_history.json')), indent=2))"
```

---

## Environment Variables (`.env`)

```
WANDB_API_KEY=your_key_here
WANDB_MODE=offline
TICKER=AAPL
FETCH_PERIOD=2y
WINDOW_SIZE=30
MODEL_BASE_PATH=./models/Qwen1.5-0.5B
OUTPUT_BASE=./output/stock
FORGET_TRIGGER=5
MIN_RETAIN_SIZE=20
POISON_SIGMA_THRESH=3.0
POISON_SWING_THRESH=0.10
POISON_VOL_MULTIPLIER=5
CYCLE_AUTO=false               # Set true for fully automatic daily cycling
PREDICTION_SAMPLES=10          # Temperature samples for confidence bands
```

---

## Colab/Kaggle Quick Start

```python
# Cell 1: Install and clone
!pip install transformers datasets accelerate yfinance wandb scikit-learn -q
!git clone https://github.com/YOUR_USERNAME/stocksense-unlearn
%cd stocksense-unlearn
!pip install -e . -q

# Cell 2: Bootstrap (download Qwen + fetch 2yr AAPL + tokenize seed)
!bash scripts/bootstrap.sh

# Cell 3: Initial fine-tune on clean 2yr AAPL seed (~10 min)
!python stocksense/training/finetune.py --config configs/finetune_config.json

# Cell 4: Inject synthetic poison + verify detector catches it
!python stocksense/data/poison_detector.py --test --inject_type all

# Cell 5: Run first super-learning cycle (~5 min on T4)
!bash scripts/run_cycle.sh

# Cell 6: Evaluate + prediction quality
!bash scripts/evaluate.sh

# Cell 7: Simulate 30 days of live ingestion with auto poison detection
!python stocksense/pipeline/ingest_loop.py --simulate --days 30

# Cell 8: Prediction on latest data
!python stocksense/prediction/predictor.py \
  --model_path output/stock/current \
  --ticker AAPL --samples 10
```

---

## Design Principles (Never Violate These)

1. **Every window is screened.** No data enters the training pipeline without passing through `poison_detector.py`. This is the first line of defense and must never be bypassed, even in testing.

2. **Unlearn before retain.** The cycle order is always: (1) erase poisoned patterns, (2) re-finetune on clean. Reversing this order means you reinforce clean patterns on top of poison — the KL term won't save you from that.

3. **The validation set is sacred.** `data/validation/held_out_clean.jsonl` is never used in any training or unlearning step. It is the only ground truth for deployment gate 3.

4. **Lower learning rate than chatbot.** Market patterns exist on a subtler signal-to-noise ratio than conversational text. 5e-6 vs 1e-5. Overly aggressive unlearning erases legitimate volatility alongside poisoned spikes.

5. **Archive poison events.** Every poisoned window is logged to `poison_log.json` with reason code and timestamp. This is your audit trail and your retrospective dataset for improving the detector.

6. **Monotonic improvement is the target.** Over N cycles: retain_ppl should trend downward (sharper clean pattern recognition), prediction_mae should trend downward, MIA_AUC should trend toward 0.5. Plot these in `cycle_history.json`. Any cycle that breaks the trend should trigger investigation before the next ingestion.

7. **Poison the test suite.** `tests/test_poison_detector.py` must include synthetic injections of every poison type. If the detector misses a synthetic flash crash, it will miss a real one.

8. **Super-learning is compound.** Each clean cycle: the model has seen more validated data AND has shed more noise. The quality improvement is not linear — it accelerates as the forget set shrinks and the retain set grows.
