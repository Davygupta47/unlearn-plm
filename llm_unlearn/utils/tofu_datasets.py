"""
Prepares tokenized TOFU datasets for unlearning experiments.
Uses forget10 as the forget set and retain90 as the retain set.
Saves all tokenized datasets to ./tokenized_dataset/tofu/

Compatible with Kaggle / Colab free tier (no large model needed at this stage).
"""
import os, sys, torch, copy
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed, BatchEncoding
from torch.utils.data import Dataset
from tqdm import trange

set_seed(42)

#Config
TOKENIZER_PATH = "Qwen/Qwen1.5-0.5B"
print("Loading tokenizer from:", TOKENIZER_PATH)
MODEL_MAX_LENGTH = 256          # reduced from 4096 for free-tier RAM
BASE_SAVE = './tokenized_dataset'

#Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    padding_side='right',
    trust_remote_code=True,
    model_max_length=MODEL_MAX_LENGTH,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#Chunking + tokenize helper
def chunk_and_tokenize(raw_dataset, max_len=MODEL_MAX_LENGTH, random_label=False):
    """
    Tokenizes a HuggingFace dataset with a 'text' column into fixed-length
    chunks, pads short chunks, and sets labels = input_ids (CLM style).
    """
    all_input_ids, all_attn, all_labels = [], [], []

    for example in raw_dataset:
        text = example['text']
        enc = tokenizer(text, truncation=False, add_special_tokens=True)
        ids  = enc['input_ids']
        mask = enc['attention_mask']

        # Chunk
        for start in range(0, len(ids), max_len):
            chunk_ids  = ids[start : start + max_len]
            chunk_mask = mask[start : start + max_len]
            if len(chunk_ids) < 5:
                continue
            pad_len = max_len - len(chunk_ids)
            chunk_ids  = chunk_ids  + [tokenizer.pad_token_id] * pad_len
            chunk_mask = chunk_mask + [0] * pad_len
            all_input_ids.append(chunk_ids)
            all_attn.append(chunk_mask)

    input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    attn_mask = torch.tensor(all_attn,      dtype=torch.long)
    #Critical fixes
    # vocab_size = tokenizer.vocab_size
    # input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    labels    = input_ids.clone()

    if random_label:
        vocab_size = tokenizer.vocab_size
        special = {tokenizer.pad_token_id, tokenizer.eos_token_id,
                   tokenizer.bos_token_id, tokenizer.unk_token_id}
        for j in range(labels.shape[0]):
            for i in range(labels.shape[1]):
                if labels[j, i].item() not in special:
                    labels[j, i] = np.random.randint(0, vocab_size - 1)

    # Mask padding in labels
    pad_mask = labels == tokenizer.pad_token_id
    labels   = torch.where(pad_mask, torch.tensor(-100), labels)

    # Build HF Dataset
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({
        'input_ids':      input_ids.tolist(),
        'attention_mask': attn_mask.tolist(),
        'labels':         labels.tolist(),
    })

#AdvSupervisedDataset (ascent+descent / KL) 
class AdvSupervisedDataset(Dataset):
    """Interleaves negative (forget) and positive (retain) examples."""
    def __init__(self, neg_ds, pos_ds, positive_factor=1.0, positive_ratio=1):
        self.input_ids, self.labels, self.attention_mask, self.factor = [], [], [], []
        pos_ids   = pos_ds['input_ids']
        pos_labs  = pos_ds['labels']
        pos_attn  = pos_ds['attention_mask']
        neg_ids   = neg_ds['input_ids']
        neg_labs  = neg_ds['labels']
        neg_attn  = neg_ds['attention_mask']

        for i in trange(len(neg_ids), desc='Building AdvDataset'):
            self.input_ids.append(neg_ids[i]);  self.labels.append(neg_labs[i])
            self.attention_mask.append(neg_attn[i]); self.factor.append(-1)
            for k in range(positive_ratio):
                idx = (i * positive_ratio + k) % len(pos_ids)
                self.input_ids.append(pos_ids[idx])
                self.labels.append(pos_labs[idx])
                self.attention_mask.append(pos_attn[idx])
                self.factor.append(positive_factor)

    def __len__(self):  return len(self.input_ids)
    def __getitem__(self, i):
        return dict(input_ids=torch.tensor(self.input_ids[i]),
                    labels=torch.tensor(self.labels[i]),
                    attention_mask=torch.tensor(self.attention_mask[i]),
                    factor=self.factor[i])

#Load TOFU splits
print('Loading TOFU dataset from HuggingFace ...')
forget_raw  = load_dataset('locuslab/TOFU', 'forget10',  split='train')
retain_raw  = load_dataset('locuslab/TOFU', 'retain90',  split='train')
full_raw    = load_dataset('locuslab/TOFU', 'full',      split='train')

def to_text(ex): return {'text': ex['question'] + '\n' + ex['answer']}
forget_raw = forget_raw.map(to_text)
retain_raw = retain_raw.map(to_text)
full_raw   = full_raw.map(to_text)

print(f'forget={len(forget_raw)}  retain={len(retain_raw)}  full={len(full_raw)}')

#Tokenize & save
def save(ds, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ds, path)
    print(f'  Saved {len(ds)} examples → {path}')

print('\nTokenizing forget set (normal) ...')
forget_ds = chunk_and_tokenize(forget_raw)
save(forget_ds, f'{BASE_SAVE}/tofu/tofu_forget/normal/tokenized_dataset.pt')

print('Tokenizing retain set (normal) ...')
retain_ds = chunk_and_tokenize(retain_raw)
save(retain_ds, f'{BASE_SAVE}/tofu/tofu_retain/normal/tokenized_dataset.pt')

print('Tokenizing full set (normal) ...')
full_ds = chunk_and_tokenize(full_raw)
save(full_ds, f'{BASE_SAVE}/tofu/tofu_full/normal/tokenized_dataset.pt')

print('Tokenizing forget set (random_label) ...')
forget_rl_ds = chunk_and_tokenize(forget_raw, random_label=True)
save(forget_rl_ds, f'{BASE_SAVE}/tofu/tofu_forget/random_label/completely_random/tokenized_dataset.pt')

print('Building ascent+descent dataset ...')
adv_ds = AdvSupervisedDataset(forget_ds, retain_ds, positive_factor=1.0, positive_ratio=1)
os.makedirs(f'{BASE_SAVE}/tofu/tofu_forget/ascent_plus_descent', exist_ok=True)
torch.save(adv_ds, f'{BASE_SAVE}/tofu/tofu_forget/ascent_plus_descent/tokenized_dataset.pt')
print(f'  Saved {len(adv_ds)} examples → ascent_plus_descent')

print('\nAll datasets tokenized and saved.')
