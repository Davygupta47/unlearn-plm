"""
run_mia.py — Membership Inference Attack evaluation.
Modified for TOFU dataset support and Kaggle/Colab free-tier.

Uses Min-K% Prob attack: measures if a sample was in training data by
checking whether the model assigns high probability to its tokens.
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
import random
import numpy as np

try:
    from transformers import is_torch_tpu_available
except ImportError:
    def is_torch_tpu_available(): return False

import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils.versions import require_version
from transformers.trainer_pt_utils import _secs2timedelta

from llm_unlearn.utils import fig_fpr_tpr

require_version("datasets>=1.8.0")
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#wandb 
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
    torch_dtype: Optional[str] = field(default=None)
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
    domain: Optional[str] = field(default=None, metadata={"help": "Domain: arxiv | github | movielens | tofu"})
    max_eval_samples: Optional[int] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)


class CustomTrainer(Trainer):
    def metrics_format(self, metrics: Dict[str, float]) -> Dict[str, float]:
        metrics_copy = metrics.copy()
        for k, v in metrics_copy.items():
            if isinstance(v, float):
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

    wandb.run.name = "mia-" + (model_args.model_name_or_path or "").replace("./output/", "", 1)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    set_seed(training_args.seed)

    #Load model 
    torch_dtype = torch.bfloat16 if _supports_tf32() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
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
        # Forget = member (label=1), retain = non-member (label=0)
        forget_dataset = torch.load(
            f"{TOKENIZED_BASE}/tofu/tofu_forget/normal/tokenized_dataset.pt",
            weights_only=False)
        approximate_dataset = torch.load(
            f"{TOKENIZED_BASE}/tofu/tofu_retain/normal/tokenized_dataset.pt",
            weights_only=False)
    elif domain == "arxiv":
        forget_dataset = torch.load(
            f"{TOKENIZED_BASE}/arxiv/arxiv_forget_500/normal/tokenized_dataset.pt",
            weights_only=False)
        approximate_dataset = torch.load(
            f"{TOKENIZED_BASE}/arxiv/arxiv_approximate_6k/normal/tokenized_dataset.pt",
            weights_only=False)
    elif domain == "github":
        forget_dataset = torch.load(
            f"{TOKENIZED_BASE}/github/github_forget_2k/normal/tokenized_dataset.pt",
            weights_only=False)
        approximate_dataset = torch.load(
            f"{TOKENIZED_BASE}/github/github_approximate/normal/tokenized_dataset.pt",
            weights_only=False)
    else:
        raise ValueError(f"Unsupported domain: {domain}")

    # Balance sizes
    n = min(len(forget_dataset), len(approximate_dataset))
    if len(approximate_dataset) > n:
        approximate_dataset = approximate_dataset.select(range(n))
    if data_args.max_eval_samples:
        n = min(n, data_args.max_eval_samples)
        forget_dataset = forget_dataset.select(range(n))
        approximate_dataset = approximate_dataset.select(range(n))

    dataset_dict = {"forget": (forget_dataset, 1), "approximate": (approximate_dataset, 0)}

    #  Min-K Prob metric 
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        labels_c = labels.clone()[:, 1:]
        pad_mask = labels_c != -100
        log_probs = torch.log_softmax(logits, dim=-1)[:, :-1]
        idx = labels_c.unsqueeze(-1).clamp(min=0)
        sel_lp = log_probs.gather(2, idx) * pad_mask.unsqueeze(-1)
        pred = logits.argmax(dim=-1)[:, :-1]
        return torch.cat((sel_lp.squeeze(-1), pred), 1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        half = preds.shape[1] // 2
        sel_lp = preds[:, :half]
        predicts = preds[:, half:]
        labels_c = labels.copy()[:, 1:]
        pad_mask = labels_c != -100
        pred_mask = predicts == labels_c

        result = {}
        for ratio in [0.3, 0.4, 0.5, 0.6, 1.0]:
            scores = []
            for lp, m, pm in zip(sel_lp, pad_mask, pred_mask):
                lp_t = torch.tensor(lp)
                nonpad_lp = lp_t[torch.tensor(m, dtype=torch.bool)]
                lp_copy = lp_t.clone()
                lp_copy[~torch.tensor(m, dtype=torch.bool)] = 100.0
                kv = max(1, int(ratio * nonpad_lp.numel()))
                topk = torch.topk(lp_copy, kv, largest=False)
                scores.append(topk.values.mean())
            result[f"min_{int(ratio*100)}_value"] = torch.stack(scores)
        return result

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if not is_torch_tpu_available() else None,
    )

    #Run mia
    if training_args.do_eval:
        os.makedirs(training_args.output_dir, exist_ok=True)
        logger.info("*** Membership Inference Attack Evaluation ***")
        all_results = []

        for key, (ds, mem_label) in dataset_dict.items():
            metrics = trainer.evaluate(ds)
            metrics_filtered = {k: v for k, v in metrics.items() if "min_" in k}
            lengths = [len(v) for v in metrics_filtered.values()]
            assert all(l == lengths[0] for l in lengths), f"Mismatched lengths: {lengths}"

            for i in range(lengths[0]):
                all_results.append({
                    "label": mem_label,
                    "pred": {k: float(v[i]) for k, v in metrics_filtered.items()},
                })

        # Shuffle for fair AUC computation
        random.seed(0)
        random.shuffle(all_results)

        output_dir = training_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        fig_fpr_tpr(all_results, output_dir)
        logger.info(f"MIA results saved to {output_dir}/auc.txt and auc.png")


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()