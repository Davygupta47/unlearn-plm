"""
Fine-tunes Qwen2-1.5B on the full TOFU dataset.
Designed for single-GPU free-tier (T4 16 GB) — uses fp16, gradient checkpointing,
small batch size + gradient accumulation.
"""
import os, sys, torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    set_seed,
)

set_seed(42)

MODEL_PATH  = os.environ.get('MODEL_PATH',  'models/Qwen1.5-0.5B')
OUTPUT_DIR  = os.environ.get('OUTPUT_DIR',  './output/tofu/Qwen1.5-0.5B/finetune')
DATA_PATH   = './tokenized_dataset/tofu/tofu_full/normal/tokenized_dataset.pt'
MAX_EPOCHS  = int(os.environ.get('FINETUNE_EPOCHS', '1'))
BATCH_SIZE  = int(os.environ.get('BATCH_SIZE', '2'))
GRAD_ACCUM  = int(os.environ.get('GRAD_ACCUM', '16'))
LR          = float(os.environ.get('LR', '2e-5'))

print(f'Fine-tuning on TOFU full set → {OUTPUT_DIR}')

# Load model
torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
    use_safetensors=True,
)
model.gradient_checkpointing_enable()  # saves ~30% VRAM

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, padding_side='right', trust_remote_code=True, model_max_length=512)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    model.resize_token_embeddings(len(tokenizer))

train_dataset = torch.load(DATA_PATH, weights_only=False)
print(f'Train samples: {len(train_dataset)}')

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=MAX_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.03,
    lr_scheduler_type='cosine',
    weight_decay=0.0,
    logging_steps=5,
    save_strategy='epoch',
    save_total_limit=1,
    fp16=(torch_dtype == torch.float32), # Use AMP fp16 for T4 (mixed precision) to speed up without crashes
    bf16=(torch_dtype == torch.bfloat16),
    report_to='none',   # disable wandb for fine-tune step
    remove_unused_columns=True,
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f'Fine-tuned model saved to {OUTPUT_DIR}')
