# Qwen-DPO Quick Guide

This directory mainly contains three parts:

- `launch.sh`: training launcher (single-GPU/multi-GPU, background run, auto-resume).
- `train_qwenimg2512_dpo.py`: main DPO training script (LoRA fine-tuning).
- `infer_qwenimage2512-tuned.py`: load trained LoRA weights and run inference.

> Note: This README describes the standard training/inference workflow only, and **intentionally ignores the `load custom_transformer` logic**.

## 1) Training (recommended via launch.sh)

```bash
cd /data/zeyu/Qwen-DPO
bash launch.sh
```

Common usage:

```bash
# Use one specific GPU
CUDA_VISIBLE_DEVICES=0 bash launch.sh

# Use multiple specific GPUs
CUDA_VISIBLE_DEVICES=2,3 bash launch.sh

# Override selected training args
bash launch.sh --max_train_steps 500 --learning_rate 1e-5
```

Training runs in the background by default, and logs are written to:

- `LOG_FILE` (default: `${OUTPUT_DIR}-nano.log`)

## 2) Auto Resume

`launch.sh` scans `OUTPUT_DIR/checkpoint-*` for valid checkpoints and automatically appends:

- `--resume_from_checkpoint latest`

To force a fresh run:

```bash
LAUNCH_NO_RESUME=1 bash launch.sh
```

## 3) Key Paths and Args

In `launch.sh`, the most frequently edited items are:

- `--pretrained_model_name_or_path` (base model directory)
- `--data_root` (DPO dataset directory)
- `--output_dir` (training output directory)
- `LEARNING_RATE` / `LR_SCHEDULER` / `LR_WARMUP_STEPS`

In `train_qwenimg2512_dpo.py`, common training args include:

- `--beta_dpo`
- `--loss_type` (`sigmoid` / `hinge` / `ipo`)
- `--lora_rank` / `--lora_alpha`
- `--checkpointing_steps` / `--logging_steps`

## 4) Inference (load trained LoRA)

Update these paths in `infer_qwenimage2512-tuned.py`:

- `model_name`: base model directory
- `finetuned_transformer_path`: your trained checkpoint (typically `.../checkpoint-xxxx/lora`)

Then run:

```bash
cd /data/zeyu/Qwen-DPO
python infer_qwenimage2512-tuned.py
```

The generated image is saved in the current directory (currently `example-tuned-new4.png` in the script).

## 5) Notes

- If you use `wandb`, make sure you are logged in and have write access.
- If VRAM is limited, reduce resolution, lower batch size, or increase gradient accumulation steps.
