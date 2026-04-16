Resource of Paper:
1. [2402.15159v3.pdf](https://github.com/user-attachments/files/26247454/2402.15159v3.pdf)
2. https://arxiv.org/pdf/2401.06121

## Execution Recipe
```console(For downloading Pre-Trained Language Model)
python -m download.py --model_name_or_path Qwen/Qwen1.5-0.5B --output_dir ./output/tofu/Qwen1.5-0.5B/finetune
```

```console(For Tokenizing Dataset)
python -m llm_unlearn.utils.tofu_datasets
```

```console(For Finetuning)
python finetune_tofu.py
```

```console(For Unlearning)
python -m llm_unlearn.run_unlearn \
  --target_model_name_or_path ./output/tofu/Qwen1.5-0.5B/finetune \
  --domain tofu --unlearn_method gradient_ascent \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 32 \
  --num_train_epochs 1 --learning_rate 1e-5 --output_dir ./output
```

```console(For Evaluation)
!python -m llm_unlearn.run_eval \
  --model_name_or_path /content/unlearn-plm/output/tofu/finetune/1_gpu_bs_1_gas_32_lr_1.0e_5_epoch1/unlearn/gradient_ascent \
  --domain tofu --do_eval --per_device_eval_batch_size 1 \
  --output_dir ./output/tofu/eval
```

```console(For Membership Inference Attack)
!python -m llm_unlearn.run_mia \
  --model_name_or_path /content/unlearn-plm/output/tofu/finetune/1_gpu_bs_1_gas_32_lr_1.0e_5_epoch1/unlearn/gradient_ascent \
  --domain tofu --do_eval --per_device_eval_batch_size 1 \
  --output_dir ./output/tofu/mia
```
