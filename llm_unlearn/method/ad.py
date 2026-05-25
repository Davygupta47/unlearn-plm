import torch
import torch.nn.functional as F
from transformers import Trainer, DataCollatorWithPadding
from torch.utils.data import SequentialSampler
from typing import Optional
import inspect
import copy

class AscentPlusDescentTrainer(Trainer):
    def __init__(self, *args, ref_model=None, kl_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if ref_model is None:
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            self.ref_model = ref_model
            self.ref_model.eval()
            
        self.kl_weight = kl_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        if "factor" not in inputs.keys():
            return super().compute_loss(model, inputs, return_outputs)
            
        factors = inputs.pop("factor") 
        labels = inputs["labels"]
        
        # 1. Forward pass of the active model undergoing unlearning
        outputs = model(**inputs)
        logits = outputs.logits
        
        # 2. Forward pass of the frozen reference model (No gradients tracked)
        with torch.no_grad():
            ref_outputs = self.ref_model(**inputs)
            ref_logits = ref_outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        # Shift operations for standard causal LM task
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Mask out non-loss tokens (like padding or prompt tokens if using a mask)
        loss_mask = shift_labels != -100
        valid_counts = loss_mask.sum(dim=-1).float()

        # Compute standard cross-entropy loss per token
        ce_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view(shift_logits.size(0), -1)
        
        # Average CE loss over valid tokens per batch item
        ce_loss = ce_loss.sum(dim=-1) / torch.clamp(valid_counts, min=1.0)

        # 3. Compute token-level KL Divergence between active and ref model
        kl_loss_fct = torch.nn.KLDivLoss(reduction="none")
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        ref_probs = F.softmax(shift_ref_logits, dim=-1)
        
        # Compute KL per token, sum over vocab, then mask non-label positions
        kl_per_token = kl_loss_fct(log_probs, ref_probs).sum(dim=-1)
        kl_per_token = kl_per_token * loss_mask
        
        # Average KL loss over valid tokens per batch item
        mean_kl_loss = kl_per_token.sum(dim=-1) / torch.clamp(valid_counts, min=1.0)

        # For forget elements (negative factor): Ascent drives up CE, while KL anchors general style.
        # For retain elements (positive factor): Descent drives down CE, matching original capabilities.
        adjusted_loss = (ce_loss * factors).mean() + self.kl_weight * mean_kl_loss.mean()
        
        return (adjusted_loss, outputs) if return_outputs else adjusted_loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)
    
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns.append('factor')


class AscentPlusDescentDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if "factor" in features[0].keys():
            batch["factor"] = torch.tensor([f["factor"] for f in features], dtype=torch.float32)
        return batch