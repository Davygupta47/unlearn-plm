"""
UnlearningArguments — extended TrainingArguments for LLM unlearning.
Adds TOFU domain, all unlearning method flags, and free-tier compatibility.
"""
import os
import transformers
try:
    from transformers import is_torch_tpu_available
except ImportError:
    def is_torch_tpu_available(): return False

from transformers import (
    TrainingArguments,
)
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UnlearningArguments(TrainingArguments):
    do_unlearn: bool = field(
        default=False, metadata={"help": "Whether to run unlearning."})
    do_unlearn_eval: bool = field(
        default=False, metadata={"help": "Whether to run unlearning eval."})

    unlearn_method: str = field(
        default="gradient_ascent",
        metadata={
            "help": (
                "Unlearning method. One of: "
                "gradient_ascent | random_label | ascent_plus_descent | "
                "ascent_plus_kl_divergence | retrain | finetune"
            )
        },
    )

    completely_random: bool = field(
        default=False,
        metadata={"help": "Use completely random labels (ignores top_k/top_p)."},
    )
    top_k: int = field(
        default=int(1e10),
        metadata={"help": "Top-k sampling for adversarial label generation."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling for adversarial label generation."},
    )
    use_soft_labels: bool = field(
        default=False,
        metadata={"help": "Use soft (distribution) labels instead of hard random labels."},
    )
    rm_groundtruth: bool = field(
        default=False,
        metadata={"help": "Remove ground-truth token when sampling adversarial labels."},
    )

    domain: str = field(
        default=None,
        metadata={
            "help": (
                "Domain to unlearn. "
                "Supported: arxiv | github | movielens | tofu"
            )
        },
    )

    general: bool = field(
        default=False,
        metadata={
            "help": (
                "Use general (out-of-domain) retain data instead of in-domain retain data "
                "for ascent_plus_descent / ascent_plus_kl_divergence. "
                "Default=False → in-domain retain data."
            )
        },
    )

    unlearned_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to an already-unlearned model (for eval-only runs)."},
    )

    # Added hyperparameter to control anchoring tightly against reference weights
    kl_weight: float = field(
        default=0.1,
        metadata={"help": "Weight coefficient for token-level KL divergence from the reference model."},
    )

    # Multiplier to amplify the gradient ascent (forget) CE loss term
    ascent_weight: float = field(
        default=3.0,
        metadata={"help": "Multiplier for gradient ascent on forget samples. Higher = stronger forgetting."},
    )

    # LoRA arguments
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for parameter-efficient unlearning."},
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha."},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout."},
    )