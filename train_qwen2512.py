"""
Qwen-Image-Edit-2511 DPO 训练脚本。

使用 Direct Preference Optimization (DPO) 训练 Qwen-Image-2512 模型，参考GitHub用户zk1009.
通过预收集的偏好对数据提升指令驱动图像编辑质量。

参考：
- DiffusionDPO (Salesforce): DPO 损失设计、训练循环结构
  https://github.com/SalesforceAIResearch/DiffusionDPO
- flow_grpo: flow matching 模型适配、LoRA 配置
  https://github.com/yifan123/flow_grpo
- Edit-R1: Qwen-Image-Edit 双通道编码逻辑

用法：
    UDA_VISIBLE_DEVICES=0 bash launch.sh
"""

import argparse
import gc
import json
import logging
import math
import os
import time
from typing import List, Optional, Union

from qwen_dpo_checkpoint import (
    initial_last_completed_batch,
    load_training_checkpoint,
    next_dataloader_position,
    resolve_resume_checkpoint,
    save_training_checkpoint,
)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
# from diffusers import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    calculate_dimensions,
)
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from datasett2itxt import DPOEditDataset

# -----------------------------------added----------------------------------------
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

logger = get_logger(__name__, log_level="INFO")

# #region agent log
_AGENT_DEBUG_LOG = "/path/to/debug.log"


def _agent_mem_log(message: str, hypothesis_id: str) -> None:
    try:
        rss_kb = None
        try:
            with open("/proc/self/status", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        break
        except OSError:
            pass
        data: dict = {
            "local_rank": os.environ.get("LOCAL_RANK", ""),
            "rss_mb": round(rss_kb / 1024.0, 1) if rss_kb is not None else None,
        }
        if torch.cuda.is_available():
            data["cuda_alloc_mb"] = round(torch.cuda.memory_allocated() / 1048576.0, 2)
            data["cuda_reserved_mb"] = round(torch.cuda.memory_reserved() / 1048576.0, 2)
        payload = {
            "sessionId": "e1ddfb",
            "hypothesisId": hypothesis_id,
            "location": "train_qwenimg2512_dpo.py:main",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG, "a", encoding="utf-8") as af:
            af.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# #endregion

CONDITION_IMAGE_AREA = 384 * 384  # Qwen2.5-VL 语义通道的图像面积

# 训练过程 infer_log 固定 prompt（t2i 对比 base vs当前 LoRA）
INFER_LOG_PROMPTS = [
    "Ultra realistic, lifelike details, natural lighting, highly detailed textures, photorealistic look. The camera focuses on a trembling hand reaching for a first aid kit in an overhead compartment. The light is harsh and unforgiving, highlighting the fear and desperation in the passenger's eyes. The cabin is in disarray, with scattered belongings and overturned seats. The scene conveys a sense of chaos and vulnerability, as the passengers struggle to cope with the unfolding crisis. The air is thick with tension and uncertainty.",
    "Ultra realistic, lifelike details, natural lighting, highly detailed textures, photorealistic look. A young child with curly dark hair, freckled skin, and tear-streaked cheeks sits barefoot on a dusty earthen floor in a dimly lit, dilapidated interior space. The child wears a soiled, short-sleeved beige shirt and dark shorts, hugging their knees with arms wrapped around shins, gazing directly at the viewer with an expression of sorrow and vulnerability. A strong diagonal beam of warm golden light enters from a high window in the upper left, illuminating dust particles in the air and casting sharp highlights on the child’s face, arms, and the floor, while the surrounding environment remains in deep shadow. Rough stone or plaster walls, indistinct furniture silhouettes, and scattered debris suggest neglect and poverty; no text is visible in the image.",
    "Ultra realistic, lifelike details, natural lighting, highly detailed textures, photorealistic look. Medium long shot, eye-level: Hana(Youth female Caucasian, long wavy brown hair, thin arched eyebrows, small nose, gentle lips, oval face; light jacket, dark pants, sneakers) sits gratefully beside Mrs. Lee(Elder female Asian, wrinkled face, kind eyes, gentle smile; simple twilight-colored shawl) on the bench. Hana is in the process of gracefully sitting down beside Mrs. Lee on the park bench. Hana shows deep gratitude and a sense of relief, while Mrs. Lee maintains her gentle, welcoming smile, as Hana carefully lowers herself onto the bench, settling in beside Mrs. Lee.",
]

INFER_LOG_NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)


# ---------------------------------------------------------------------------
# 编码函数（参考 Edit-R1 的 qwen_image_edit_pipeline_with_logprob.py）
# ---------------------------------------------------------------------------

def _get_qwen_prompt_embeds(
    pipeline,
    prompt: Union[str, List[str]],
    image=None,
    device=None,
    dtype=None,
    max_seq_len: int = 512,
):
    """
    通过 Qwen2.5-VL 文本编码器编码 (源图像 + 编辑指令)。

    参考：Edit-R1 的 _get_qwen_prompt_embeds 实现。
    """
    device = device or pipeline._execution_device
    dtype = dtype or pipeline.text_encoder.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt

    img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
    base_img_prompt_list = []
    if isinstance(image, list):
        for i, img in enumerate(image):
            base_img_prompt_list.append(img_prompt_template.format(i + 1))
    elif image is not None:
        base_img_prompt_list.append(img_prompt_template.format(1))
    else:
        base_img_prompt_list = [""] * len(prompt)

    template = pipeline.prompt_template_encode
    drop_idx = pipeline.prompt_template_encode_start_idx
    txt = [
        template.format(base_img + p)
        for base_img, p in zip(base_img_prompt_list, prompt)
    ]

    model_inputs = pipeline.processor(
        text=txt,
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(device)

    outputs = pipeline.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = pipeline._extract_masked_hidden(
        hidden_states, model_inputs.attention_mask
    )
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [
        torch.ones(e.size(0), dtype=torch.long, device=e.device)
        for e in split_hidden_states
    ]

    prompt_embeds = torch.stack([
        torch.cat([
            u[:max_seq_len] if u.size(0) > max_seq_len else u,
            u.new_zeros(max(0, max_seq_len - u.size(0)), u.size(1)),
        ])
        for u in split_hidden_states
    ])
    encoder_attention_mask = torch.stack([
        torch.cat([
            u[:max_seq_len] if u.size(0) > max_seq_len else u,
            u.new_zeros(max(0, max_seq_len - u.size(0))),
        ])
        for u in attn_mask_list
    ])

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return prompt_embeds, encoder_attention_mask

def encode_target_images(pipeline, images, device, dtype, resolution):
    """
    通过 VAE 编码目标图像（优选或劣选），返回打包后的 latents (x0)。

    参考：QwenImageEditPlusPipeline 的 _encode_vae_image() + _pack_latents()
    """
    vae_image_area = resolution * resolution
    num_channels_latents = pipeline.transformer.config.in_channels // 4

    latents_list = []
    for img in images:
        image_width, image_height = img.size
        vae_w, vae_h = calculate_dimensions(
            vae_image_area, image_width / image_height
        )
        processed = pipeline.image_processor.preprocess(img, vae_h, vae_w).unsqueeze(2)
        processed = processed.to(device=device, dtype=torch.float32)
        latent = pipeline._encode_vae_image(image=processed, generator=None)
        h_lat, w_lat = latent.shape[3:]
        packed = pipeline._pack_latents(
            latent, 1, num_channels_latents, h_lat, w_lat
        )
        packed = packed.to(dtype=dtype, device=device)
        latents_list.append(packed)

    return torch.cat(latents_list, dim=0)


# ---------------------------------------------------------------------------
# DPO 损失函数
# ---------------------------------------------------------------------------

def compute_dpo_loss(
    transformer,
    x0_pref,            # [B, seq, C] 优选图像打包 latents
    x0_rej,             # [B, seq, C] 劣选图像打包 latents
    prompt_embeds,      # [B, txt_seq, D] 文本+语义编码
    prompt_embeds_mask, # [B, txt_seq]
    img_shapes,         # list of shape tuples
    txt_seq_lens,       # list of ints
    beta_dpo,           # DPO 温度
    loss_type,          # "sigmoid" / "hinge" / "ipo"
):
    """
    Flow matching DPO 损失。

    参考：
    - DiffusionDPO: logits = ref_diff - model_diff → logsigmoid 损失
    - flow_grpo train_sd3_dpo.py: flow matching 前向过程 xt = (1-t)*x0 + t*noise
    - Edit-R1 train_nft_qwen_image_edit.py line 1045-1099: transformer 前向调用方式

    公式：
        xt = (1-t) * x0 + t * ε
        v_target = ε - x0
        MSE = ||v_pred - v_target||²
        logits = (MSE_ref_w - MSE_ref_l) - (MSE_pi_w - MSE_pi_l)
        L_DPO = -log σ(β * logits)
    """
    device = x0_pref.device
    batch_size = x0_pref.shape[0]

    # 1. 采样时间步 t ~ U(0.001, 0.999) 和共享噪声
    t = torch.rand(batch_size, device=device) * 0.998 + 0.001
    noise = torch.randn_like(x0_pref)  # 优选和劣选共享同一噪声

    # 2. Flow matching 前向加噪：xt = (1-t) * x0 + t * noise
    t_exp = t.view(-1, 1, 1)
    xt_pref = (1 - t_exp) * x0_pref + t_exp * noise
    xt_rej = (1 - t_exp) * x0_rej + t_exp * noise

    # 3. 拼接源图像 latents（外观通道）
    xt_pref_input = torch.cat([xt_pref], dim=1)
    xt_rej_input = torch.cat([xt_rej], dim=1)

    # 4. 拼接优选+劣选为一个 batch（减少前向次数：4→2）
    #    参考 DiffusionDPO 的 feed_pixel_values = cat(chunk(2)) 技巧
    xt_combined = torch.cat([xt_pref_input, xt_rej_input], dim=0)
    t_combined = t.repeat(2)
    prompt_combined = prompt_embeds.repeat(2, 1, 1)
    mask_combined = prompt_embeds_mask.repeat(2, 1)
    img_shapes_combined = img_shapes + img_shapes
    txt_seq_combined = txt_seq_lens + txt_seq_lens

    seq_len_target = xt_pref.shape[1]

    # 5. 策略模型前向（LoRA 启用）
    transformer.enable_adapters()
    v_pi = transformer(
        hidden_states=xt_combined,
        timestep=t_combined,
        encoder_hidden_states=prompt_combined,
        encoder_hidden_states_mask=mask_combined, #[2,512]
        img_shapes=img_shapes_combined,
        txt_seq_lens=txt_seq_combined, # [512,512]
        guidance=None,
        return_dict=False,
    )[0][:, :seq_len_target]
    v_pi_pref, v_pi_rej = v_pi.chunk(2)

    # 6. 参考模型前向（LoRA 禁用，no_grad）
    with torch.no_grad():
        transformer.disable_adapters()
        v_ref = transformer(
            hidden_states=xt_combined,
            timestep=t_combined,
            encoder_hidden_states=prompt_combined,
            encoder_hidden_states_mask=mask_combined,
            img_shapes=img_shapes_combined,
            txt_seq_lens=txt_seq_combined,
            guidance=None,
            return_dict=False,
        )[0][:, :seq_len_target]
        v_ref_pref, v_ref_rej = v_ref.chunk(2)
        transformer.enable_adapters()

    # 7. 计算真实速度目标：v_target = noise - x0
    target_pref = noise - x0_pref
    target_rej = noise - x0_rej

    # 8. 逐样本 MSE（在序列和通道维度上取均值）
    mse_pi_pref = ((v_pi_pref - target_pref) ** 2).mean(dim=(1, 2))
    mse_pi_rej = ((v_pi_rej - target_rej) ** 2).mean(dim=(1, 2))
    mse_ref_pref = ((v_ref_pref - target_pref) ** 2).mean(dim=(1, 2))
    mse_ref_rej = ((v_ref_rej - target_rej) ** 2).mean(dim=(1, 2))

    # 9. DPO 损失
    # 参考 DiffusionDPO: logits = ref_diff - model_diff
    model_diff = mse_pi_pref - mse_pi_rej
    ref_diff = mse_ref_pref - mse_ref_rej
    logits = ref_diff - model_diff

    if loss_type == "sigmoid":
        loss = -F.logsigmoid(beta_dpo * logits).mean()
    elif loss_type == "hinge":
        loss = torch.relu(1 - beta_dpo * logits).mean()
    elif loss_type == "ipo":
        loss = ((logits - 1 / (2 * beta_dpo)) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # 指标
    with torch.no_grad():
        implicit_acc = (logits > 0).float().mean()
        reward_pref = -(mse_pi_pref - mse_ref_pref)
        reward_rej = -(mse_pi_rej - mse_ref_rej)
        reward_margin = (reward_pref - reward_rej).mean()
        beta_logits_abs_mean = (beta_dpo * logits).abs().mean()

    return loss, {
        "loss": loss.detach(),
        "implicit_acc": implicit_acc,
        "mse_pi_pref": mse_pi_pref.mean().detach(),
        "mse_pi_rej": mse_pi_rej.mean().detach(),
        "mse_ref_pref": mse_ref_pref.mean().detach(),
        "mse_ref_rej": mse_ref_rej.mean().detach(),
        "reward_margin": reward_margin,
        "logits_mean": logits.mean().detach(),
        "beta_logits_abs_mean": beta_logits_abs_mean.detach(),
    }


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DPO training for Qwen-Image-Edit")

    # 模型
    parser.add_argument(
        "--pretrained_model_name_or_path", type=str,
        default="Qwen/Qwen-Image-2512",
        help="预训练模型路径或 HuggingFace model ID",
    )
    parser.add_argument("--resolution", type=int, default=1024)

    # DPO
    parser.add_argument(
        "--beta_dpo", type=float, default=500.0,
        help="DPO 温度参数（参考 DiffusionDPO SDXL: 5000）",
    )
    parser.add_argument(
        "--loss_type", type=str, default="sigmoid",
        choices=["sigmoid", "hinge", "ipo"],
    )

    # LoRA
    parser.add_argument(
        "--lora_rank", type=int, default=64,
        help="LoRA 秩（参考 flow_grpo: 32）",
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=128,
        help="LoRA alpha（参考 flow_grpo: 64）",
    )

    # 训练
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--lr_scheduler", type=str, default="constant",
        choices=["constant", "linear", "cosine"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=100)

    # 混合精度
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--allow_tf32", action="store_true", default=True)

    # 数据
    parser.add_argument("--data_root", type=str, required=False, default="/path/to/your/dataset")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--max_sequence_length", type=int, default=512)

    # 输出
    parser.add_argument("--output_dir", type=str, default="output/dpo")
    parser.add_argument("--checkpointing_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--report_to", type=str, default="wandb",
        choices=["wandb", "tensorboard", "none"],
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # 恢复训练：checkpoint 目录，或 "latest" 使用 output_dir 下最新 checkpoint-*
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--debug_log_steps",
        type=int,
        default=10,
        help="每隔多少个 global step 打印一次梯度/参数更新诊断日志",
    )

    # 训练过程对比推理（base 关 LoRA vs 开 LoRA），与 checkpoint 周期对齐便于对照
    parser.add_argument(
        "--infer_log_steps",
        type=int,
        default=200,
        help="每隔多少 global step 做一次 infer_log；0 表示关闭",
    )
    parser.add_argument(
        "--infer_log_disable",
        action="store_true",
        help="关闭 infer_log（优先级高于 infer_log_steps）",
    )
    parser.add_argument("--infer_num_inference_steps", type=int, default=50)
    parser.add_argument("--infer_log_cfg_scale", type=float, default=4.0)
    parser.add_argument("--infer_log_width", type=int, default=1664)
    parser.add_argument("--infer_log_height", type=int, default=928)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# infer_log：base（disable_adapters）与当前 LoRA（enable_adapters）同 seed 对比
# ---------------------------------------------------------------------------


def _infer_log_font(size: int = 20):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _compose_infer_log_pair(img_base: Image.Image, img_lora: Image.Image) -> Image.Image:
    """左右拼接，底部标注 base model / lora。"""
    if img_base.height != img_lora.height:
        h = max(img_base.height, img_lora.height)
        if img_base.height != h:
            img_base = img_base.resize(
                (int(img_base.width * h / img_base.height), h), Image.Resampling.LANCZOS
            )
        if img_lora.height != h:
            img_lora = img_lora.resize(
                (int(img_lora.width * h / img_lora.height), h), Image.Resampling.LANCZOS
            )
    w1, w2 = img_base.width, img_lora.width
    caption_h = 52
    canvas = Image.new("RGB", (w1 + w2, img_base.height + caption_h), (245, 245, 245))
    canvas.paste(img_base, (0, 0))
    canvas.paste(img_lora, (w1, 0))
    draw = ImageDraw.Draw(canvas)
    font = _infer_log_font(20)
    y_text = img_base.height + 14
    for text, cx in (("base model", w1 // 2), ("lora", w1 + w2 // 2)):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, y_text), text, fill=(25, 25, 25), font=font)
    return canvas


def infer_log_should_run(args: argparse.Namespace, global_step: int) -> bool:
    if getattr(args, "infer_log_disable", False):
        return False
    if args.infer_log_steps <= 0:
        return False
    return global_step % args.infer_log_steps == 0


def run_infer_log(
    accelerator: Accelerator,
    pipeline,
    transformer: torch.nn.Module,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    """仅主进程：对 INFER_LOG_PROMPTS 各生成 base vs lora 对比图。"""
    if not accelerator.is_main_process:
        return
    unwrapped = accelerator.unwrap_model(transformer)
    out_dir = os.path.join(args.output_dir, "infer_log", f"step_{global_step:06d}")
    os.makedirs(out_dir, exist_ok=True)
    was_training = transformer.training
    transformer.eval()
    unwrapped.eval()
    device = accelerator.device
    try:
        # 使用 no_grad 而非 inference_mode：后者产生的 inference tensor 不能与后续
        # gradient checkpoint + autograd 共存（会报 Inference tensors cannot be saved for backward）。
        with torch.no_grad():
            for i, prompt in enumerate(INFER_LOG_PROMPTS):
                seed = int(args.seed) + i * 1000 + int(global_step)
                unwrapped.disable_adapters()
                g0 = torch.Generator(device=device).manual_seed(seed)
                img_base = pipeline(
                    prompt=prompt,
                    negative_prompt=INFER_LOG_NEGATIVE_PROMPT,
                    width=args.infer_log_width,
                    height=args.infer_log_height,
                    num_inference_steps=args.infer_num_inference_steps,
                    true_cfg_scale=args.infer_log_cfg_scale,
                    generator=g0,
                ).images[0].convert("RGB")
                unwrapped.enable_adapters()
                g1 = torch.Generator(device=device).manual_seed(seed)
                img_lora = pipeline(
                    prompt=prompt,
                    negative_prompt=INFER_LOG_NEGATIVE_PROMPT,
                    width=args.infer_log_width,
                    height=args.infer_log_height,
                    num_inference_steps=args.infer_num_inference_steps,
                    true_cfg_scale=args.infer_log_cfg_scale,
                    generator=g1,
                ).images[0].convert("RGB")
                combined = _compose_infer_log_pair(img_base, img_lora)
                combined.save(os.path.join(out_dir, f"prompt_{i:02d}.png"))
        logger.info("infer_log 已写入 %s", out_dir)
    finally:
        unwrapped.enable_adapters()
        if was_training:
            transformer.train()
        else:
            transformer.eval()


def maybe_infer_log(
    accelerator: Accelerator,
    pipeline,
    transformer: torch.nn.Module,
    global_step: int,
    args: argparse.Namespace,
) -> None:
    if not infer_log_should_run(args, global_step):
        return
    run_infer_log(accelerator, pipeline, transformer, global_step, args)
    accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Accelerator 初始化 ---
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to if args.report_to != "none" else None,
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- 加载模型 ---
    logger.info(f"Loading pipeline from {args.pretrained_model_name_or_path}")
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
    )

    # 冻结所有组件
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)

    # VAE 保持 fp32
    pipeline.vae.to(dtype=torch.float32)
    transformer = pipeline.transformer
    
    # vae,text_encoder,transformer 移到设备
    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device)
    transformer = pipeline.transformer.to(accelerator.device)

    # --- LoRA 设置 ---
    # 参考 flow_grpo 的 LoRA target modules
    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_out.0",
        "attn.to_q",
        "attn.to_v",
    ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_config)
    logger.info(
        f"LoRA applied: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"trainable params: {sum(p.numel() for p in transformer.parameters() if p.requires_grad):,}"
    )

    # 梯度检查点
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # --- 优化器 ---
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # --- 数据集 ---
    train_dataset = DPOEditDataset(
        args.data_root, split="train", resolution=args.resolution
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=DPOEditDataset.collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- 学习率调度器 ---
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = min(
        args.max_train_steps,
        args.num_train_epochs * num_update_steps_per_epoch,
    )

    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    resume_ckpt_path = None
    if args.resume_from_checkpoint:
        resume_ckpt_path = resolve_resume_checkpoint(
            args.resume_from_checkpoint, args.output_dir
        )

    # --- Accelerate 准备 ---
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler,
    )

    resume_epoch_st = 0
    resume_batch_idx_st = 0
    global_step = 0
    if resume_ckpt_path:
        global_step, resume_epoch_st, resume_batch_idx_st = load_training_checkpoint(
            resume_ckpt_path,
            accelerator,
            transformer,
            optimizer,
            lr_scheduler,
        )
        logger.info(
            "Resumed from %s global_step=%s next_pos=(epoch=%s, batch=%s)",
            resume_ckpt_path,
            global_step,
            resume_epoch_st,
            resume_batch_idx_st,
            main_process_only=True,
        )

    len_per_epoch = len(train_dataloader)
    last_epoch_for_ckpt, last_step_for_ckpt = initial_last_completed_batch(
        resume_epoch_st, resume_batch_idx_st, len_per_epoch
    )

    # 诊断：仅跟踪可训练参数（主要是 LoRA），用于判断是否真的发生更新
    def _collect_trainable_named_params():
        named = []
        for n, p in accelerator.unwrap_model(transformer).named_parameters():
            if p.requires_grad:
                named.append((n, p))
        return named

    tracked_named_params = _collect_trainable_named_params()
    prev_param_snapshot = {
        n: p.detach().float().cpu().clone()
        for n, p in tracked_named_params
    }

    # --- 初始化日志 ---
    if accelerator.is_main_process and args.report_to != "none":
        run_name = args.run_name or f"dpo-qwen-edit-{args.resolution}"
        accelerator.init_trackers(
            project_name="qwen-image-2512-tuned-0403",
            config=vars(args),
            init_kwargs={"wandb": {"name": "qwen-dpo-97"}},
        )

    # --- 训练循环 ---
    logger.info("***** 开始训练 *****")
    logger.info(f"  数据集大小 = {len(train_dataset)}")
    logger.info(f"  每 GPU batch size = {args.train_batch_size}")
    logger.info(f"  梯度累积步数 = {args.gradient_accumulation_steps}")
    logger.info(
        f"  有效 batch size = "
        f"{args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
    )
    logger.info(f"  最大训练步数 = {max_train_steps}")
    logger.info(f"  DPO beta = {args.beta_dpo}")
    logger.info(f"  损失类型 = {args.loss_type}")

    progress_bar = tqdm(
        range(max_train_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
        initial=global_step,
    )

    maybe_infer_log(accelerator, pipeline, transformer, global_step, args)

    for epoch in range(resume_epoch_st, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            if epoch == resume_epoch_st and step < resume_batch_idx_st:
                continue
            with accelerator.accumulate(transformer):
                # 1. 编码源图像 + 指令（no_grad，冻结组件）
                with torch.no_grad():
                    prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                        prompt=batch["instruction"],
                        prompt_embeds=None,
                        prompt_embeds_mask=None,
                        device=accelerator.device,
                        num_images_per_prompt=len(batch["instruction"]),
                        max_sequence_length=512,
                    )
                    # 每个样本各自一个 txt_seq_len
                    txt_seq_lens = [prompt_embeds_mask.size(1)] * len(batch["instruction"])
                    


                    def _load_rgb_image(image_path: str) -> Image.Image:
                            return Image.open(image_path).convert("RGB")


                    def _center_crop_and_resize(img_rgb: Image.Image, resolution: int) -> Image.Image:
                        """center crop 到正方形，再 resize 到指定分辨率。"""
                        w, h = img_rgb.size
                        if w != h:
                            min_dim = min(w, h)
                            left = (w - min_dim) // 2
                            top = (h - min_dim) // 2
                            img_rgb = img_rgb.crop((left, top, left + min_dim, top + min_dim))

                        return img_rgb.resize((resolution, resolution), Image.Resampling.LANCZOS)


                    def _load_or_generate_image_pair(
                        preferred_path: str,
                        rejected_path: str,
                        instruction: str,
                        resolution: int,
                        pipeline,
                        device: str = "cuda",
                    ) -> tuple[Image.Image, Image.Image]:
                        """
                        读取 preferred image；
                        如果 rejected_path 为空，则先按 preferred 原图尺寸生成 rejected image；
                        最后对两者统一做 crop + resize。
                        """
                        # 1) 先加载 preferred 原图（注意：这里先不 resize）
                        preferred_rgb = _load_rgb_image(preferred_path)

                        # 2) rejected: 有路径就加载；没路径就现场生成
                        if rejected_path and rejected_path.strip():
                            rejected_rgb = _load_rgb_image(rejected_path)
                        else:
                            rejected_rgb = pipeline(
                                prompt=instruction,
                                negative_prompt=(
                                    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
                                    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
                                ),
                                width=preferred_rgb.width,
                                height=preferred_rgb.height,
                                num_inference_steps=50,
                                true_cfg_scale=4.0,
                                generator=torch.Generator(device=accelerator.device).manual_seed(42),
                            ).images[0].convert("RGB")

                        # 3) 再统一 crop + resize
                        preferred_img = _center_crop_and_resize(preferred_rgb, resolution)
                        rejected_img = _center_crop_and_resize(rejected_rgb, resolution)

                        # preferred_img.save("preferred_img.png")
                        # rejected_img.save("rejected_img.png")
                        return preferred_img, rejected_img


                    # ===== 批处理 =====
                    preferred_images = []
                    rejected_images = []

                    for instruction, preferred_path, rejected_path in zip(
                        batch["instruction"],
                        batch["preferred_image"],
                        batch["rejected_image"],
                    ):
                        preferred_img, rejected_img = _load_or_generate_image_pair(
                            preferred_path=preferred_path,
                            rejected_path=rejected_path,
                            instruction=instruction,
                            resolution=args.resolution,
                            pipeline=pipeline,
                            device="cuda",
                        )
                        preferred_images.append(preferred_img)
                        rejected_images.append(rejected_img)

                    # 最终结果
                    prefered_image = preferred_images
                    rejected_image = rejected_images
                    
                    # prefered_image[0].save("prefered_image_rgb1.png")
                    # rejected_image[0].save("rejected_image_rgb1.png")

                    x0_pref = encode_target_images(
                        pipeline, prefered_image,
                        accelerator.device, torch.bfloat16, args.resolution,
                    )
                    x0_rej = encode_target_images(
                        pipeline, rejected_image,
                        accelerator.device, torch.bfloat16, args.resolution,
                    )

                # h_l = prefered_image[0].height // 16
                # w_l = prefered_image[0].width // 16
                # img_shapes = [[(1, w_l, h_l)]]
                # 每个样本各自一个 img_shape
                img_shapes = [[(1, 64, 64)] for _ in range(len(batch["instruction"]))]
                # 2. 计算 DPO 损失
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                loss, metrics = compute_dpo_loss(
                    unwrapped_transformer,
                    x0_pref, x0_rej,
                    prompt_embeds, prompt_embeds_mask,
                    img_shapes, txt_seq_lens,
                    args.beta_dpo, args.loss_type,
                )

                # 3. 反向传播
                accelerator.backward(loss)

                # 诊断：统计梯度范数与非零梯度参数占比
                grad_sq_sum = torch.tensor(0.0, device=accelerator.device)
                nonzero_grad_params = 0
                total_grad_params = 0
                for _, p in tracked_named_params:
                    if p.grad is None:
                        continue
                    g = p.grad.detach().float()
                    total_grad_params += 1
                    grad_sq_sum += torch.sum(g * g)
                    if torch.any(g != 0):
                        nonzero_grad_params += 1
                grad_norm = torch.sqrt(grad_sq_sum)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

                # 诊断：统计参数步进量（本 step 更新前后差异）
                delta_sq_sum = torch.tensor(0.0, device=accelerator.device)
                param_sq_sum = torch.tensor(0.0, device=accelerator.device)
                for n, p in tracked_named_params:
                    curr = p.detach().float()
                    prev = prev_param_snapshot[n].to(curr.device)
                    d = curr - prev
                    delta_sq_sum += torch.sum(d * d)
                    param_sq_sum += torch.sum(curr * curr)
                    prev_param_snapshot[n] = curr.cpu().clone()
                param_delta_norm = torch.sqrt(delta_sq_sum)
                param_norm = torch.sqrt(param_sq_sum)
                param_update_ratio = param_delta_norm / (param_norm + 1e-12)

                optimizer.zero_grad()

            # --- 日志和检查点 ---
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)

                maybe_infer_log(accelerator, pipeline, transformer, global_step, args)

                if global_step % args.logging_steps == 0:
                    log_dict = {
                        "train/loss": metrics["loss"].item(),
                        "train/implicit_acc": metrics["implicit_acc"].item(),
                        "train/mse_pi_pref": metrics["mse_pi_pref"].item(),
                        "train/mse_pi_rej": metrics["mse_pi_rej"].item(),
                        "train/reward_margin": metrics["reward_margin"].item(),
                        "train/logits_mean": metrics["logits_mean"].item(),
                        "train/beta_logits_abs_mean": metrics["beta_logits_abs_mean"].item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/grad_norm": grad_norm.detach().item(),
                        "train/nonzero_grad_params": float(nonzero_grad_params),
                        "train/total_grad_params": float(total_grad_params),
                        "train/param_delta_norm": param_delta_norm.detach().item(),
                        "train/param_update_ratio": param_update_ratio.detach().item(),
                    }
                    accelerator.log(log_dict, step=global_step)
                    if accelerator.is_main_process:
                        logger.info(
                            f"step={global_step}, loss={metrics['loss'].item():.4f}, "
                            f"acc={metrics['implicit_acc'].item():.3f}, "
                            f"margin={metrics['reward_margin'].item():.4f}, "
                            f"grad_norm={grad_norm.detach().item():.3e}, "
                            f"upd_ratio={param_update_ratio.detach().item():.3e}, "
                            f"nz_grad={nonzero_grad_params}/{total_grad_params}"
                        )

                if (
                    accelerator.is_main_process
                    and global_step % args.debug_log_steps == 0
                ):
                    logger.info(
                        "[debug] logits_mean=%.6f, mse_pi_pref=%.6f, mse_pi_rej=%.6f, "
                        "mse_ref_pref=%.6f, mse_ref_rej=%.6f, beta_logits_abs=%.6f",
                        metrics["logits_mean"].item(),
                        metrics["mse_pi_pref"].item(),
                        metrics["mse_pi_rej"].item(),
                        metrics["mse_ref_pref"].item(),
                        metrics["mse_ref_rej"].item(),
                        metrics["beta_logits_abs_mean"].item(),
                    )

                if (
                    global_step > 0
                    and global_step % args.checkpointing_steps == 0
                ):
                    ne, nb = next_dataloader_position(epoch, step, len_per_epoch)
                    save_training_checkpoint(
                        accelerator,
                        transformer,
                        optimizer,
                        lr_scheduler,
                        args.output_dir,
                        global_step,
                        ne,
                        nb,
                    )

                if global_step >= max_train_steps:
                    last_epoch_for_ckpt = epoch
                    last_step_for_ckpt = step
                    break

            last_epoch_for_ckpt = epoch
            last_step_for_ckpt = step

        if global_step >= max_train_steps:
            break

    # --- 保存最终模型 ---
    ne, nb = next_dataloader_position(
        last_epoch_for_ckpt, last_step_for_ckpt, len_per_epoch
    )
    save_training_checkpoint(
        accelerator,
        transformer,
        optimizer,
        lr_scheduler,
        args.output_dir,
        global_step,
        ne,
        nb,
    )
    if accelerator.is_main_process:
        logger.info(f"训练完成！最终模型保存在 {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
