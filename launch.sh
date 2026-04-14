#!/bin/bash
# Qwen-Image-Edit-2511 DPO 训练启动脚本
#
# 用法：
#   CUDA_VISIBLE_DEVICES=0 bash launch.sh              # 只用0号卡
#   bash launch.sh                                    # 默认使用所有GPU
#   bash launch.sh --max_train_steps 500              # 传递额外参数
#
# 断点续训：若 OUTPUT_DIR 下已有新格式 checkpoint（training_state.bin +
# lora/adapter_model.safetensors），会自动加上 --resume_from_checkpoint latest。
# 从头重新训练请设置：LAUNCH_NO_RESUME=1 bash launch.sh
# 或改用新的 OUTPUT_DIR；命令行里后出现的 --resume_from_checkpoint 会覆盖上面的 latest
#
# 日志：统一写入 LOG_FILE（默认 ${OUTPUT_DIR}-nano.log），开头会打印学习率等关键超参，
#       便于与训练过程日志在同一文件中对照。

# 如果需要, 这里可以自定义要用的 GPU，例如:
# export CUDA_VISIBLE_DEVICES=0

# 与下方 train 参数保持一致（用于自动续训检测与日志头）
OUTPUT_DIR="${OUTPUT_DIR:-output-nano-0410/dpo}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}-nano.log}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
LR_SCHEDULER="${LR_SCHEDULER:-constant}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-100}"

# 新格式 checkpoint 才可 resume；仅存在旧版大 checkpoint 时不自动附加，避免启动报错
RESUME_ARGS=()
if [ -z "${LAUNCH_NO_RESUME:-}" ]; then
    best_step=-1
    shopt -s nullglob
    for d in "${OUTPUT_DIR}"/checkpoint-*; do
        [ -d "$d" ] || continue
        if [[ -f "$d/training_state.bin" && -f "$d/lora/adapter_model.safetensors" ]]; then
            suffix="${d##*checkpoint-}"
            if [[ "$suffix" =~ ^[0-9]+$ ]] && ((10#$suffix > best_step)); then
                best_step=$((10#$suffix))
            fi
        fi
    done
    shopt -u nullglob
    if ((best_step >= 0)); then
        RESUME_ARGS=(--resume_from_checkpoint latest)
        echo "[launch.sh] Auto-resume: found new-format checkpoint (latest step ${best_step}) under ${OUTPUT_DIR}" >&2
    fi
fi

launch_log_header() {
    echo "======== launch.sh $(date -Iseconds 2>/dev/null || date) ========"
    echo "LOG_FILE=${LOG_FILE}"
    echo "OUTPUT_DIR=${OUTPUT_DIR}"
    echo "learning_rate=${LEARNING_RATE}  lr_scheduler=${LR_SCHEDULER}  lr_warmup_steps=${LR_WARMUP_STEPS}"
    echo "Note: trailing \"\$@\" may override matching CLI flags (e.g. --learning_rate)."
    if ((${#RESUME_ARGS[@]} > 0)); then
        echo "RESUME_ARGS=${RESUME_ARGS[*]}"
    fi
    echo "--------------------------------------------"
}

# 自动获取可见设备数量作为进程数
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # 未手动指定, 默认2卡
    NUM_GPUS=${NUM_GPUS:-2}
else
    # 根据逗号分割数量推断
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    # 单卡，不用 --multi_gpu, 只指定1个进程
    {
        launch_log_header
        accelerate launch \
            --num_processes=1 \
            --mixed_precision=bf16 \
            train_qwenimg2512_dpo.py \
            --pretrained_model_name_or_path="/data/Qwen-Image-2512" \
            --data_root="/data/zeyu/Qwen-DPO/dataset/doubao-10k-nano" \
            --output_dir="${OUTPUT_DIR}" \
            --resolution=1024 \
            --beta_dpo=500 \
            --loss_type="sigmoid" \
            --lora_rank=64 \
            --lora_alpha=128 \
            --learning_rate="${LEARNING_RATE}" \
            --train_batch_size=1 \
            --gradient_accumulation_steps=2 \
            --max_train_steps=5000 \
            --lr_scheduler="${LR_SCHEDULER}" \
            --lr_warmup_steps="${LR_WARMUP_STEPS}" \
            --gradient_checkpointing \
            --mixed_precision=bf16 \
            --checkpointing_steps=200 \
            --logging_steps=10 \
            --report_to=wandb \
            --seed=42 \
            "${RESUME_ARGS[@]}" \
            "$@"
    } > "${LOG_FILE}" 2>&1 &
else
    # 多卡，使用 --multi_gpu 参数
    {
        launch_log_header
        accelerate launch \
            --multi_gpu \
            --num_processes="${NUM_GPUS}" \
            --mixed_precision=bf16 \
            train_qwenimg2512_dpo.py \
            --pretrained_model_name_or_path="/data/Qwen-Image-2512" \
            --data_root="/data/zeyu/Qwen-DPO/dataset/doubao-10k-nano" \
            --output_dir="${OUTPUT_DIR}" \
            --resolution=1024 \
            --beta_dpo=500 \
            --loss_type="sigmoid" \
            --lora_rank=64 \
            --lora_alpha=128 \
            --learning_rate="${LEARNING_RATE}" \
            --train_batch_size=1 \
            --gradient_accumulation_steps=2 \
            --max_train_steps=5000 \
            --lr_scheduler="${LR_SCHEDULER}" \
            --lr_warmup_steps="${LR_WARMUP_STEPS}" \
            --gradient_checkpointing \
            --mixed_precision=bf16 \
            --checkpointing_steps=200 \
            --logging_steps=10 \
            --report_to=wandb \
            --seed=42 \
            "${RESUME_ARGS[@]}" \
            "$@"
    } > "${LOG_FILE}" 2>&1 &
fi

echo "[launch.sh] Training started in background; log: ${LOG_FILE}" >&2
