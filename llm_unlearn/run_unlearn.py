"""
run_unlearn.py — LLM Unlearning entry point.
Modified for:
  - TOFU dataset support (forget10 / retain90)
  - Qwen1.5-0.5B compatibility
  - Kaggle / Colab free-tier (single T4, fp16, no FSDP required)
  - wandb optional (falls back to offline/disabled silently)
"""

import logging
import os
import sys
import warnings
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset

try:
    from transformers import is_torch_tpu_available
except ImportError:
    def is_torch_tpu_available(): return False

import transformers
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from llm_unlearn.method import (
    GradientAscentTrainer,
    UnlearningArguments,
    AscentPlusKLDivergenceTrainer,
    AscentPlusDescentDataCollator,
    AscentPlusDescentTrainer,
)
from llm_unlearn.utils import (
    load_model_and_tokenizer,
)

require_version("datasets>=1.8.0")

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

#wandb
try:
    import wandb as _wm
    _KEY = os.environ.get("WANDB_API_KEY", "").strip()
    if _KEY:
        _wm.login(key=_KEY)
    else:
        os.environ.setdefault("WANDB_MODE", "offline")
        _wm.login(anonymous="allow")
    import wandb
    wandb.init(project="LLMUnlearn")
    _WANDB_OK = True
except Exception as _e:
    print(f"[wandb] Disabled: {_e}")
    _WANDB_OK = False
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


# Argument dataclasses

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    target_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Fine-tuned model to unlearn."})
    trust_remote_code: bool = field(default=True)
    model_max_length: int = field(default=512)

    def __post_init__(self):
        pass


@dataclass
class DataTrainingArguments:
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    dataset_name: Optional[str] = field(default=None)
    dataset_config_name: Optional[str] = field(default=None)


#  Domain to dataset path map 

DOMAIN_PATHS = {
    "arxiv":     "arxiv/arxiv_forget_500",
    "github":    "github/github_forget_2k",
#   "movielens": "movielens/movielens_forget_500",
    "tofu":      "tofu/tofu_forget",
}

TOKENIZED_BASE = "./tokenized_dataset"


def _get_dataset_path(domain: str, sub: str) -> str:
    """Return full path to a tokenized dataset file."""
    return os.path.join(TOKENIZED_BASE, DOMAIN_PATHS[domain], sub, "tokenized_dataset.pt")


def main():
    # Strip --tf32 on pre-Ampere / CPU
    def _supports_tf32():
        if not torch.cuda.is_available(): return False
        return torch.cuda.get_device_capability(0)[0] >= 8

    if not _supports_tf32():
        sys.argv = [a for a in sys.argv if a != "--tf32" and not a.startswith("--tf32=")]

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UnlearningArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        json_path = os.path.abspath(sys.argv[1])
        with open(json_path) as f:
            data = json.load(f)
        if not _supports_tf32():
            data["tf32"] = False
        model_args, data_args, training_args = parser.parse_dict(data)
    else:
        parsed = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        model_args, data_args, training_args, remaining = parsed
        if remaining:
            # Handle --overwrite_output_dir flag gracefully
            if "--overwrite_output_dir" in remaining:
                remaining = [a for a in remaining if a != "--overwrite_output_dir"]
                training_args.overwrite_output_dir = True
            if remaining and remaining != ["True"] and remaining != ["False"]:
                raise ValueError(f"Unknown args: {remaining}")

    set_seed(training_args.seed)

    # Disable intermediate checkpoints to prevent disk space exhaustion (e.g. on Kaggle)
    training_args.save_strategy = "no"

    # Enable AMP fp16 on T4 (pre-Ampere) for ~3x training speedup.
    # Model weights stay in float32; Trainer handles mixed precision internally.
    if not _supports_tf32():
        training_args.fp16 = True
    else:
        training_args.bf16 = True

    # Build output path
    lr_str = "{:.1e}".format(training_args.learning_rate).replace("-0", "_").replace("-", "_")
    path = model_args.model_name_or_path or model_args.target_model_name_or_path
    model_name = os.path.basename(os.path.normpath(path)) if path else "model"
    n_gpu = max(1, torch.cuda.device_count())

    overall_output_dir = os.path.join(
        training_args.output_dir if training_args.output_dir else "./output",
        f"{training_args.domain}",
        f"{model_name}",
        f"{n_gpu}_gpu_bs_{training_args.per_device_train_batch_size}"
        f"_gas_{training_args.gradient_accumulation_steps}"
        f"_lr_{lr_str}_epoch{int(training_args.num_train_epochs)}",
    )
    if training_args.general:
        overall_output_dir += "general"
    if training_args.rm_groundtruth:
        overall_output_dir += "_rmgt"

    method = training_args.unlearn_method

    if method == "random_label":
        if training_args.completely_random:
            prefix = "random_label-completely_random"
        elif training_args.use_soft_labels:
            prefix = "random_label-soft_label"
        else:
            if training_args.top_k == int(1e10):
                prefix = f"random_label-top_p{int(training_args.top_p*100)}"
            elif training_args.top_p == 1:
                prefix = f"random_label-top_k{training_args.top_k}"
            else:
                prefix = f"random_label-top_k{training_args.top_k}_top_p{training_args.top_p}"
    else:
        prefix = method

    training_args.output_dir = os.path.join(overall_output_dir, "unlearn", prefix)
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Logging
    log_dir = training_args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, "run.log")),
        ],
    )
    logger.setLevel(logging.INFO)
    transformers.utils.logging.set_verbosity_info()

    logger.info(f"Method: {method} | Domain: {training_args.domain}")
    logger.info(f"Output: {training_args.output_dir}")

    # Determine fine-tuned model path
    finetuned_path = (
        model_args.target_model_name_or_path
        or os.path.join(os.path.dirname(os.path.dirname(training_args.output_dir)), "finetune")
    )
    pretrained_path = model_args.model_name_or_path

    Trainer_args = {"args": training_args}

    # TOFU dataset paths
    domain = training_args.domain
    if domain not in DOMAIN_PATHS:
        raise ValueError(f"Unsupported domain: {domain}. Choose from {list(DOMAIN_PATHS)}")

    # Build unlearner
    if method == "retrain":
        model, tokenizer = load_model_and_tokenizer(pretrained_path)
        train_dataset = torch.load(
            _get_dataset_path(domain, "normal"), weights_only=False)
        unlearner = Trainer(model=model, train_dataset=train_dataset, **Trainer_args)

    elif method == "finetune":
        model, tokenizer = load_model_and_tokenizer(finetuned_path)
        train_dataset = torch.load(
            _get_dataset_path(domain, "normal"), weights_only=False)
        unlearner = Trainer(model=model, train_dataset=train_dataset,**Trainer_args)

    elif method == "random_label":
        model, tokenizer = load_model_and_tokenizer(finetuned_path)
        if training_args.completely_random:
            ds_sub = "random_label/completely_random"
        else:
            sub = f"top_k{int(training_args.top_k)}_top_p{training_args.top_p}"
            if training_args.rm_groundtruth:
                sub += "_rmgt"
            ds_sub = f"random_label/{sub}"
        train_dataset = torch.load(
            _get_dataset_path(domain, ds_sub), weights_only=False)
        if data_args.max_train_samples:
            train_dataset = train_dataset.select(range(
                min(len(train_dataset), data_args.max_train_samples)))
        unlearner = Trainer(model=model, train_dataset=train_dataset, **Trainer_args)

    elif method == "gradient_ascent":
        model, tokenizer = load_model_and_tokenizer(finetuned_path)
        train_dataset = torch.load(
            _get_dataset_path(domain, "normal"), weights_only=False)
        if data_args.max_train_samples:
            train_dataset = train_dataset.select(range(
                min(len(train_dataset), data_args.max_train_samples)))
        unlearner = GradientAscentTrainer(
            model=model, train_dataset=train_dataset, **Trainer_args)

    elif method in ("ascent_plus_descent", "ascent_plus_kl_divergence"):
        model, tokenizer = load_model_and_tokenizer(finetuned_path)
        ds_sub = "ascent_plus_descent_general" if training_args.general else "ascent_plus_descent"
        train_dataset = torch.load(
            _get_dataset_path(domain, ds_sub), weights_only=False)
        if data_args.max_train_samples:
            train_dataset = train_dataset.select(range(
                min(len(train_dataset), data_args.max_train_samples)))

        if method == "ascent_plus_descent":
            unlearner = AscentPlusDescentTrainer(
                model=model, train_dataset=train_dataset,
                **Trainer_args,
                data_collator=AscentPlusDescentDataCollator(tokenizer),
            )
        else:  # ascent_plus_kl_divergence
            params = {
                "torch_dtype": torch.bfloat16 if _supports_tf32() else torch.float32,
                "trust_remote_code": True,
            }
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                finetuned_path, **params)
            unlearner = AscentPlusKLDivergenceTrainer(
                pretrain_model=pretrained_model,
                model=model, train_dataset=train_dataset,
                **Trainer_args,
                data_collator=AscentPlusDescentDataCollator(tokenizer),
            )
    else:
        raise ValueError(f"Unknown unlearn_method: {method}")

    #Train 
    logger.info(f"Starting unlearning with method={method} on domain={domain}")
    t0 = time.time()
    print("Model vocab:", model.config.vocab_size)
    sample = train_dataset[0]["input_ids"]
    print("Max token id:", max(sample))
    result = unlearner.train()
    elapsed = time.time() - t0
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Unlearning done in {int(h)}h {int(m)}m {s:.1f}s")

    unlearner.save_model()
    metrics = result.metrics
    metrics["train_samples"] = len(unlearner.train_dataset)
    unlearner.log_metrics("train", metrics)
    unlearner.save_metrics("train", metrics)
    # Note: save_state() intentionally omitted to avoid writing optimizer/scheduler
    # state to disk, which can exhaust limited Kaggle/Colab storage.
    logger.info(f"Model saved to {training_args.output_dir}")


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()