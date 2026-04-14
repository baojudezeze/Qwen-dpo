import os
import json
import torch
from peft import LoraConfig, set_peft_model_state_dict
from safetensors.torch import load_file
from diffusers import DiffusionPipeline


model_name = "Qwen/Qwen-Image-2512"
finetuned_transformer_path = "/path/to/checkpoint/lora"

if torch.cuda.is_available():
    dtype = torch.bfloat16
    device = "cuda"
else:
    dtype = torch.float32
    device = "cpu"


def load_sharded_safetensors(ckpt_dir):
    index_file = os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors.index.json")
    with open(index_file, "r", encoding="utf-8") as f:
        index_json = json.load(f)

    weight_map = index_json["weight_map"]
    shard_files = sorted(set(weight_map.values()))

    state_dict = {}
    for shard in shard_files:
        shard_path = os.path.join(ckpt_dir, shard)
        shard_sd = load_file(shard_path, device="cpu")
        state_dict.update(shard_sd)

    return state_dict


def merge_peft_lora_into_base_state_dict(peft_sd, lora_scale=1.0, target_dtype=torch.float32):
    """
    把这种 key:
      xxx.base_layer.weight
      xxx.base_layer.bias
      xxx.lora_A.default.weight
      xxx.lora_B.default.weight
    合并成普通模型可加载的:
      xxx.weight
      xxx.bias
    """

    merged_sd = {}
    lora_A = {}
    lora_B = {}

    for k, v in peft_sd.items():
        # base layer -> 普通权重名
        if ".base_layer." in k:
            new_k = k.replace(".base_layer", "")
            merged_sd[new_k] = v.to(target_dtype)

        # LoRA A
        elif ".lora_A.default.weight" in k:
            prefix = k.replace(".lora_A.default.weight", "")
            lora_A[prefix] = v.to(torch.float32)

        # LoRA B
        elif ".lora_B.default.weight" in k:
            prefix = k.replace(".lora_B.default.weight", "")
            lora_B[prefix] = v.to(torch.float32)

        else:
            # 其他普通参数直接保留
            merged_sd[k] = v.to(target_dtype)

    # 合并 LoRA: W <- W_base + scale * (B @ A)
    common_prefixes = set(lora_A.keys()) & set(lora_B.keys())
    for prefix in common_prefixes:
        weight_key = prefix + ".weight"
        A = lora_A[prefix]   # [r, in]
        B = lora_B[prefix]   # [out, r]
        delta = torch.matmul(B, A) * lora_scale  # [out, in]

        if weight_key in merged_sd:
            base_w = merged_sd[weight_key].to(torch.float32)
            merged_sd[weight_key] = (base_w + delta).to(target_dtype)
        else:
            # 极少数情况下没有 base_layer.weight，就直接用 delta
            merged_sd[weight_key] = delta.to(target_dtype)

    return merged_sd


def resolve_lora_directory(path: str) -> str:
    """Directory that contains weights: either .../lora or .../checkpoint-N/lora (new), or legacy shard dir."""
    if os.path.isfile(os.path.join(path, "adapter_model.safetensors")):
        return path
    sub = os.path.join(path, "lora")
    if os.path.isfile(os.path.join(sub, "adapter_model.safetensors")):
        return sub
    if os.path.isfile(os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")):
        return path
    if os.path.isfile(os.path.join(sub, "diffusion_pytorch_model.safetensors.index.json")):
        return sub
    return path


def load_peft_lora_adapter(transformer, lora_dir: str) -> None:
    """标准 PEFT LoRA：adapter_config.json + adapter_model.safetensors（与训练保存一致）。"""
    cfg = LoraConfig.from_pretrained(lora_dir)
    transformer.add_adapter(cfg)
    weights_path = os.path.join(lora_dir, "adapter_model.safetensors")
    sd = load_file(weights_path, device="cpu")
    set_peft_model_state_dict(transformer, sd, adapter_name="default")
    transformer.enable_adapters()
    print(f"Loaded PEFT LoRA from {lora_dir} ({len(sd)} tensors)")


def load_finetuned_transformer_into_pipe(pipe, ckpt_dir, lora_scale=1.0):
    adapter_cfg_path = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        r = adapter_cfg.get("r", None)
        alpha = adapter_cfg.get("lora_alpha", None)
        if r and alpha:
            lora_scale = lora_scale * (alpha / r)
            print(f"Using effective LoRA scale = {lora_scale:.6f} (alpha/r applied)")

    # 1) 读 PEFT/LoRA 风格的 sharded checkpoint
    peft_sd = load_sharded_safetensors(ckpt_dir)

    # 2) 合并成普通 transformer 可识别的 state_dict
    merged_sd = merge_peft_lora_into_base_state_dict(
        peft_sd,
        lora_scale=lora_scale,
        target_dtype=dtype,
    )

    # 3) 加载到当前 pipeline 的 transformer
    missing, unexpected = pipe.transformer.load_state_dict(merged_sd, strict=False)

    print("=== load transformer done ===")
    print(f"missing keys: {len(missing)}")
    print(f"unexpected keys: {len(unexpected)}")

    if len(missing) > 0:
        print("sample missing keys:", missing[:20])
    if len(unexpected) > 0:
        print("sample unexpected keys:", unexpected[:20])

    if len(missing) == 0 and len(unexpected) == 0:
        print("State dict key match looks good.")


def load_finetuned_lora_into_pipe(pipe, user_path: str, lora_scale: float = 1.0) -> None:
    lora_dir = resolve_lora_directory(user_path)
    if os.path.isfile(os.path.join(lora_dir, "adapter_model.safetensors")):
        if lora_scale != 1.0:
            print("Warning: lora_scale != 1.0 is not applied for PEFT-native load; adjust training alpha/r if needed.")
        load_peft_lora_adapter(pipe.transformer, lora_dir)
    elif os.path.isfile(os.path.join(lora_dir, "diffusion_pytorch_model.safetensors.index.json")):
        load_finetuned_transformer_into_pipe(pipe, lora_dir, lora_scale=lora_scale)
    else:
        raise FileNotFoundError(
            f"No adapter_model.safetensors or diffusion_pytorch_model.safetensors.index.json under {lora_dir}"
        )


# ---------------------------
# 先加载 base pipeline
# 关键：关闭 low_cpu_mem_usage，避免 meta tensor 问题
# ---------------------------
pipe = DiffusionPipeline.from_pretrained(
    model_name,
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
)

# 第二步：加载 DPO 后的 LoRA（新格式：PEFT adapter；旧格式：整模 shard + merge）
if os.path.isdir(finetuned_transformer_path):
    load_finetuned_lora_into_pipe(pipe, finetuned_transformer_path, lora_scale=1.0)
else:
    raise FileNotFoundError(
        f"Finetuned LoRA checkpoint not found: {finetuned_transformer_path}"
    )

# 最后再整体放到 device
pipe = pipe.to(device)

print(type(pipe.transformer))
print(pipe.transformer.config._class_name)
print(pipe.transformer.config.num_layers)




# Generate image
prompt = '''A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm.'''
negative_prompt = "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1104),
    "3:4": (1104, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cuda").manual_seed(42)
).images[0]

image.save("example-tuned-new4.png")
