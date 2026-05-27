import torch
import torch.nn.functional as F
from transformers import Trainer
from torch.utils.data import SequentialSampler
from typing import Optional
import inspect

class AscentPlusKLDivergenceTrainer(Trainer):
    def __init__(self, pretrain_model=None, kl_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        device = self.accelerator.device
        if pretrain_model is not None:
            pretrain_model.to(device)
            pretrain_model.eval()
            for param in pretrain_model.parameters():
                param.requires_grad = False
        self.pretrain_model = pretrain_model
        self.kl_weight = kl_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "factor" not in inputs.keys():
            return super().compute_loss(model, inputs, return_outputs, **kwargs)
        
        factors = inputs.pop("factor")
        labels = inputs["labels"]
        device = self.accelerator.device

        # 1. Main forward pass for the entire batch (both retain and forget data)
        outputs = model(**inputs)
        logits = outputs.logits

        # 2. Reference forward pass for the entire batch (no gradient tracking)
        with torch.no_grad():
            pretrained_outputs = self.pretrain_model(**inputs)
            ref_logits = pretrained_outputs.logits

        # Causal shift adjustments for standard text generation tasks
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Create our loss calculation masks (ignoring padding/prompts marked -100)
        loss_mask = shift_labels != -100
        valid_counts = loss_mask.sum(dim=-1).float()

        # PART A: Cross-Entropy Loss (Modulated by Forget/Retain Factors) ----
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        ce_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view(shift_logits.size(0), -1)
        
        # Average cross entropy over the target tokens for each sequence in the batch
        ce_loss_per_seq = ce_per_token.sum(dim=-1) / torch.clamp(valid_counts, min=1.0)
        # Apply the factors dynamically (+1 for regular descent, -1 for gradient ascent)
        adjusted_ce_loss = (ce_loss_per_seq * factors).mean()

        # ---- PART B: Global Forward-KL Regularization Anchor ----
        kl_loss_fct = torch.nn.KLDivLoss(reduction="none")
        log_probs = F.log_softmax(shift_logits, dim=-1)
        ref_probs = F.softmax(shift_ref_logits, dim=-1)

        kl_per_token = kl_loss_fct(log_probs, ref_probs).sum(dim=-1)
        kl_per_token = kl_per_token * loss_mask # Zero-out masked positions
        mean_kl_loss = kl_per_token.sum(dim=-1) / torch.clamp(valid_counts, min=1.0)
        
        # Combine the task loss with our structural anchor constraint
        total_loss = adjusted_ce_loss + (self.kl_weight * mean_kl_loss.mean())

        return (total_loss, outputs) if return_outputs else total_loss

    def _get_train_sampler(self, dataset: Optional[torch.utils.data.Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(dataset if dataset is not None else self.train_dataset)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(
                set(["label", "label_ids"] + self.label_names)
            )
            self._signature_columns.append("factor")