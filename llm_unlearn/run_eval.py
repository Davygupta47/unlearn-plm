"""
run_eval.py — Evaluation script for unlearned LLMs.
Modified for TOFU dataset support and Kaggle/Colab free-tier.

Evaluates:
  - forget set: perplexity + token accuracy (want HIGH ppl = model forgot)
  - general / retain set: perplexity + token accuracy (want LOW ppl = model retained)
"""

import logging
import math
import os
import sys
import warnings
import json
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch

try:
    from transformers import is_torch_tpu_available
except ImportError:
    def is_torch_tpu_available(): return False

import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils.versions import require_version
try:
    from transformers.trainer_pt_utils import _secs2timedelta
except ImportError:
    def _secs2timedelta(secs):
        h, rem = divmod(int(secs), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

require_version("datasets>=1.8.0")
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#Optional wandb
try:
    import wandb as _wm
    os.environ.setdefault("WANDB_MODE", "offline")
    _wm.login(anonymous="allow")
    import wandb
    wandb.init(project="LLMUnlearn")
except Exception:
    class _WandbShim:
        class run:
            name = ""
        @staticmethod
        def log(*a, **kw): pass
        @staticmethod
        def login(*a, **kw): pass
        @staticmethod
        def init(*a, **kw): pass
    wandb = _WandbShim()


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    trust_remote_code: bool = field(default=True)
    torch_dtype: Optional[str] = field(default=None, metadata={"choices": ["auto", "bfloat16", "float16", "float32"]})
    low_cpu_mem_usage: bool = field(default=False)
    model_max_length: int = field(default=512)
    use_auth_token: Optional[bool] = field(default=None)
    token: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    model_revision: str = field(default="main")
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    config_overrides: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None)
    def __post_init__(self):
        pass


@dataclass
class DataTrainingArguments:
    domain: Optional[str] = field(default=None, metadata={"help": "Unlearned domain: arxiv | github | movielens | tofu"})
    max_eval_samples: Optional[int] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)


class CustomTrainer(Trainer):
    def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if "_mem_" in k:
                metrics_copy[k] = f"{v >> 20}MB"
            elif "_runtime" in k:
                metrics_copy[k] = _secs2timedelta(v)
            elif k == "total_flos":
                metrics_copy[k] = f"{int(v) >> 30}GF"
            elif isinstance(metrics_copy[k], float):
                metrics_copy[k] = round(v, 4)
        return metrics_copy


def main():
    def _supports_tf32():
        if not torch.cuda.is_available(): return False
        return torch.cuda.get_device_capability(0)[0] >= 8

    if not _supports_tf32():
        sys.argv = [a for a in sys.argv if a != "--tf32" and not a.startswith("--tf32=")]

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    wandb.run.name = "eval-" + (model_args.model_name_or_path or "").replace("./output/", "", 1)

    # Logging     
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    set_seed(training_args.seed)

    # Load model 
    torch_dtype = (
        getattr(torch, model_args.torch_dtype)
        if model_args.torch_dtype and model_args.torch_dtype not in ["auto", None]
        else (torch.bfloat16 if _supports_tf32() else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        trust_remote_code=model_args.trust_remote_code,
    )
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # Load datasets     
    domain = data_args.domain
    TOKENIZED_BASE = "./tokenized_dataset"

    if domain == "tofu":
        forget_dataset = torch.load(
            f"{TOKENIZED_BASE}/tofu/tofu_forget/normal/tokenized_dataset.pt",
            weights_only=False)
        retain_dataset = torch.load(
            f"{TOKENIZED_BASE}/tofu/tofu_retain/normal/tokenized_dataset.pt",
            weights_only=False)
        dataset_dict = {"forget": forget_dataset, "retain": retain_dataset}
    elif domain == "arxiv":
        forget_dataset = torch.load(
            f"{TOKENIZED_BASE}/arxiv/arxiv_forget_500/normal/tokenized_dataset.pt",
            weights_only=False)
        general_dataset = torch.load(
            f"{TOKENIZED_BASE}/general/general_1k/normal/tokenized_dataset.pt",
            weights_only=False)
        dataset_dict = {"forget": forget_dataset, "general": general_dataset}
    elif domain == "github":
        forget_dataset = torch.load(
            f"{TOKENIZED_BASE}/github/github_forget_2k/normal/tokenized_dataset.pt",
            weights_only=False)
        general_dataset = torch.load(
            f"{TOKENIZED_BASE}/general/general_1k/normal/tokenized_dataset.pt",
            weights_only=False)
        dataset_dict = {"forget": forget_dataset, "general": general_dataset}
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    # Metrics helpers     
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        labels_cloned = labels.clone()[:, 1:]
        pad_mask = labels_cloned != -100
        log_probs = torch.log_softmax(logits, dim=-1)[:, :-1]
        idx = labels_cloned.unsqueeze(-1).clamp(min=0)
        sel_log_probs = log_probs.gather(2, idx) * pad_mask.unsqueeze(-1)
        pred = logits.argmax(dim=-1)[:, :-1]
        return torch.cat((sel_log_probs.squeeze(-1), pred), 1)

    def compute_min_k_ppl_acc(sel_lp, mask, k, pred_mask):
        ppls, accs = [], []
        for lp, m, pm in zip(sel_lp, mask, pred_mask):
            lp_nonpad = lp[m]
            lp_copy = lp.clone()  # FIX: changed from .copy() to .clone()
            lp_copy[~m] = 100.0
            kv = max(1, int(k * lp_nonpad.numel()))
            topk = torch.topk(lp_copy, kv, largest=False)  # FIX: removed torch.tensor() wrapper
            avg_lp = topk.values.mean()
            ppls.append(avg_lp)
            accs.append(pm[topk.indices].float().mean())
        avg_lp = torch.stack(ppls).mean()
        ppl = torch.exp(-avg_lp).item()
        acc = (sum(accs) / len(accs) * 100).item() if accs else 0.0
        return ppl, acc

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # FIX: Trainer converts to numpy arrays. Cast back to tensors.
        preds = torch.as_tensor(preds)
        labels = torch.as_tensor(labels)
        
        half = preds.shape[1] // 2
        sel_lp = preds[:, :half]
        predicts = preds[:, half:]
        labels_c = labels.clone()[:, 1:]  # FIX: changed from .copy() to .clone()
        mask = labels_c != -100
        pred_mask = predicts == labels_c
        ppl, acc = compute_min_k_ppl_acc(sel_lp, mask, 1.0, pred_mask)
        return {"ppl": ppl, "acc": acc}

    # Trainer     
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics if not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if not is_torch_tpu_available() else None,
    )

    # Evaluate 
    if training_args.do_eval:
        os.makedirs(training_args.output_dir, exist_ok=True)
        logger.info("*** Evaluate ***")
        summary = {}
        for key, ds in dataset_dict.items():
            if data_args.max_eval_samples:
                ds = ds.select(range(min(len(ds), data_args.max_eval_samples)))
            metrics = trainer.evaluate(ds)
            try:
                metrics["perplexity"] = math.exp(metrics["eval_loss"])
            except OverflowError:
                metrics["perplexity"] = float("inf")
            metrics["eval_samples"] = len(ds)
            trainer.log_metrics(f"{key}_eval", metrics)
            trainer.save_metrics(f"{key}_eval", metrics)
            summary[key] = {k: v for k, v in metrics.items() if k in ("eval_loss", "perplexity", "eval_ppl", "eval_acc")}

        print("\n===== Evaluation Summary =====")
        for split, m in summary.items():
            print(f"  [{split}]  ppl={m.get('perplexity', m.get('eval_ppl', '?')):.2f}  "
                  f"loss={m.get('eval_loss', '?'):.4f}")
        print("==============================\n")


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
