#!/usr/bin/env python3
"""Vehicle recognition with Qwen3-VL + cross-modal attention overlay.

Uses the smallest official Qwen3-VL instruct checkpoint (2B). There is no
official Qwen3-VL-0.6B; Qwen3-0.6B is text-only.

Attention maps come from HuggingFace eager attention (not vLLM PagedAttention).
Run inside conda env `vllm` (PyTorch cu128 / sm_120 for RTX 5060 Ti).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT / ".deps"

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# transformers>=4.57 lives in .deps so the vllm env's older transformers stay intact.
if LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))

import transformers.utils.import_utils as _hf_import_utils

_hf_import_utils._accelerate_available = False

import torch


DEFAULT_PROMPT = (
    "这是一张道路场景图。请只列出图中实际看得见的车辆，不要提及没有出现的车型。"
    "对每一辆车用中文分条写：类型、颜色、画面位置（左/中/右，近/中/远）、行驶或停靠。"
    "不要描述行人、交通灯、建筑或道路。"
)

VEHICLE_KEYWORDS = (
    "车",
    "车辆",
    "卡车",
    "货车",
    "罐车",
    "公交",
    "轿车",
    "皮卡",
    "工程车",
    "面包",
    "SUV",
    "suv",
    "vehicle",
    "truck",
    "bus",
    "car",
    "van",
    "tanker",
    "pickup",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT / "assets" / "traffic_tanker_light.jpg",
        help="Input traffic image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "models" / "Qwen3-VL-2B-Instruct"),
        help="Local model directory or HuggingFace / ModelScope id",
    )
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=512 * 28 * 28,
        help="Cap vision tokens so eager attention fits in 16GB",
    )
    parser.add_argument("--min-pixels", type=int, default=128 * 28 * 28)
    parser.add_argument(
        "--last-layers",
        type=int,
        default=8,
        help="Average attention over the last N language-model layers",
    )
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap blend weight")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    return parser.parse_args()


def setup_local_transformers() -> None:
    import transformers

    print(f"[info] transformers={transformers.__version__} from {transformers.__file__}")
    print(f"[info] torch={torch.__version__} cuda={torch.version.cuda} file={torch.__file__}")
    if torch.cuda.is_available():
        print(f"[info] gpu={torch.cuda.get_device_name(0)} arch={torch.cuda.get_arch_list()}")
    try:
        from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "Failed to import Qwen3VLForConditionalGeneration. "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def load_model(model_path: str, min_pixels: int, max_pixels: int):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    print(f"[info] loading {model_path}")
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_path)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None:
            if hasattr(image_processor, "size"):
                image_processor.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
            if hasattr(image_processor, "min_pixels"):
                image_processor.min_pixels = min_pixels
                image_processor.max_pixels = max_pixels

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )
    model.to("cuda")
    model.eval()
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    return model, processor


def build_inputs(processor, image_path: Path, prompt: str, device: torch.device):
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    kwargs = dict(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    if video_inputs:
        kwargs["videos"] = video_inputs
    inputs = processor(**kwargs)
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


@torch.inference_mode()
def generate_text(model, processor, inputs: dict, max_new_tokens: int) -> tuple[torch.Tensor, str]:
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        use_cache=True,
    )
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_len:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return generated, text.strip()


def _as_attn_tensor(layer_attn) -> torch.Tensor:
    if isinstance(layer_attn, (tuple, list)):
        layer_attn = layer_attn[0]
    return layer_attn


@torch.inference_mode()
def collect_attentions(model, inputs: dict, generated_ids: torch.Tensor):
    prompt_len = inputs["input_ids"].shape[1]
    extra = generated_ids.shape[1] - prompt_len
    attn_mask = inputs["attention_mask"]
    if extra > 0:
        pad = torch.ones((attn_mask.shape[0], extra), device=attn_mask.device, dtype=attn_mask.dtype)
        attn_mask = torch.cat([attn_mask, pad], dim=1)

    vision_kwargs = {k: v for k, v in inputs.items() if k not in {"input_ids", "attention_mask"}}
    outputs = model(
        input_ids=generated_ids,
        attention_mask=attn_mask,
        output_attentions=True,
        return_dict=True,
        **vision_kwargs,
    )
    if outputs.attentions is None:
        raise RuntimeError("Model returned no attentions. Eager attention is required.")
    return outputs.attentions, prompt_len


def vision_token_index(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    idx = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise RuntimeError(f"No image tokens (id={image_token_id}) found in the prompt.")
    return idx


def llm_grid(image_grid_thw: torch.Tensor, spatial_merge_size: int) -> tuple[int, int, int]:
    grid = image_grid_thw[0].tolist()
    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
    return t, h // spatial_merge_size, w // spatial_merge_size


def mean_query_to_vision(
    attentions,
    query_index: torch.Tensor | int,
    vision_idx: torch.Tensor,
    last_layers: int,
) -> torch.Tensor:
    """Average heads and selected layers: query token(s) -> vision tokens."""
    layers = [_as_attn_tensor(a) for a in attentions]
    chosen = layers[-last_layers:] if last_layers > 0 else layers
    if isinstance(query_index, int):
        q = torch.tensor([query_index], device=chosen[0].device)
    else:
        q = query_index.to(chosen[0].device)

    acc = None
    for layer_attn in chosen:
        # layer_attn: (batch, heads, q_len, k_len)
        gathered = layer_attn[0, :, q][:, :, vision_idx].float()
        layer_mean = gathered.mean(dim=(0, 1))
        acc = layer_mean if acc is None else acc + layer_mean
    return acc / len(chosen)


def reshape_heatmap(vec: torch.Tensor, t: int, gh: int, gw: int) -> np.ndarray:
    n = t * gh * gw
    if vec.numel() != n:
        raise RuntimeError(f"Vision attention length {vec.numel()} != grid {t}x{gh}x{gw}={n}")
    heat = vec.reshape(t, gh, gw).mean(dim=0).detach().cpu().float().numpy()
    return heat


def overlay_heatmap(image_bgr: np.ndarray, heat: np.ndarray, alpha: float) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(w, h) * 0.012)
    lo, hi = np.percentile(heat, 5), np.percentile(heat, 99)
    heat = np.clip((heat - lo) / (hi - lo + 1e-8), 0, 1)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, color, alpha, 0)


def _cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Prefer a single-file CJK fallback; Noto TTC index 0 is Japanese and misses many Hans glyphs.
    candidates = (
        ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
        ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
    )
    for path, index in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size, index=index)
    return ImageFont.load_default()


def _draw_label(img_bgr: np.ndarray, text: str, xy: tuple[int, int] = (16, 12)) -> np.ndarray:
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = _cjk_font(28)
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 8, bbox[1] - 6, bbox[2] + 8, bbox[3] + 6), fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def decode_generated_pieces(processor, token_ids: torch.Tensor) -> list[tuple[int, str]]:
    pieces = []
    for i, tid in enumerate(token_ids.tolist()):
        piece = processor.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pieces.append((i, piece))
    return pieces


def pick_keyword_queries(pieces: list[tuple[int, str]], prompt_len: int) -> list[tuple[int, str]]:
    picked = []
    seen = set()
    for rel_i, piece in pieces:
        token = piece.strip()
        if not token:
            continue
        if any(key.lower() in token.lower() or token.lower() in key.lower() for key in VEHICLE_KEYWORDS):
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append((prompt_len + rel_i, token))
    return picked[:6]


def _resize_max_width(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    nh = int(h * max_w / w)
    return cv2.resize(img, (max_w, nh), interpolation=cv2.INTER_AREA)


def make_panel(original_bgr: np.ndarray, tiles: list[tuple[str, np.ndarray]], title: str) -> np.ndarray:
    tiles = [("原图", original_bgr)] + tiles
    tiles = [(label, _draw_label(_resize_max_width(img, 1400), label)) for label, img in tiles]
    h, w = tiles[0][1].shape[:2]
    cols = min(2, len(tiles))
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, (_label, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        vis = img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h))
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = vis

    banner = np.full((88, canvas.shape[1], 3), 20, dtype=np.uint8)
    banner = _draw_label(banner, title[:80], (16, 24))
    return np.vstack([banner, canvas])


def save_caption(path: Path, text: str) -> None:
    font = _cjk_font(22)
    img = Image.new("RGB", (1600, 520), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    draw.multiline_text((24, 24), text, fill=(240, 240, 240), font=font, spacing=8)
    img.save(path)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    setup_local_transformers()
    model, processor = load_model(args.model, args.min_pixels, args.max_pixels)
    device = next(model.parameters()).device

    image = Image.open(args.image).convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    inputs = build_inputs(processor, args.image, args.prompt, device)
    generated_ids, answer = generate_text(model, processor, inputs, args.max_new_tokens)
    print("\n===== vehicle recognition =====\n" + answer + "\n")

    attentions, prompt_len = collect_attentions(model, inputs, generated_ids)
    image_token_id = getattr(model.config, "image_token_id", 151655)
    vision_idx = vision_token_index(generated_ids, image_token_id)
    merge = model.config.vision_config.spatial_merge_size
    t, gh, gw = llm_grid(inputs["image_grid_thw"], merge)
    print(f"[info] vision tokens={vision_idx.numel()}  llm_grid=({t},{gh},{gw})  prompt_len={prompt_len}")

    gen_rel = torch.arange(prompt_len, generated_ids.shape[1], device=generated_ids.device)
    if gen_rel.numel() == 0:
        raise RuntimeError("Model generated no new tokens.")

    overall = mean_query_to_vision(attentions, gen_rel, vision_idx, args.last_layers)
    overall_heat = reshape_heatmap(overall, t, gh, gw)
    overall_overlay = overlay_heatmap(image_bgr, overall_heat, args.alpha)

    pieces = decode_generated_pieces(processor, generated_ids[0, prompt_len:])
    keyword_queries = pick_keyword_queries(pieces, prompt_len)
    tiles = [("车辆生成token → 图像", overall_overlay)]
    keyword_meta = []
    for abs_i, token in keyword_queries:
        vec = mean_query_to_vision(attentions, abs_i, vision_idx, args.last_layers)
        heat = reshape_heatmap(vec, t, gh, gw)
        overlay = overlay_heatmap(image_bgr, heat, args.alpha)
        tiles.append((f"token「{token}」", overlay))
        token_path = args.out_dir / f"attn_token_{token.strip() or abs_i}.jpg"
        cv2.imwrite(str(token_path), overlay)
        keyword_meta.append({"token": token, "index": abs_i, "path": str(token_path)})

    stem = args.image.stem
    overlay_path = args.out_dir / f"{stem}_attention_overlay.jpg"
    panel_path = args.out_dir / f"{stem}_attention_panel.jpg"
    caption_path = args.out_dir / f"{stem}_caption.png"
    json_path = args.out_dir / f"{stem}_result.json"
    raw_heat_path = args.out_dir / f"{stem}_heatmap.npy"

    cv2.imwrite(str(overlay_path), overall_overlay)
    cv2.imwrite(str(panel_path), make_panel(image_bgr, tiles, f"Qwen3-VL-2B 车辆 attention | {stem}"))
    save_caption(caption_path, "Qwen3-VL-2B-Instruct\n\n" + answer)
    np.save(raw_heat_path, overall_heat)

    result = {
        "model": args.model,
        "image": str(args.image),
        "prompt": args.prompt,
        "answer": answer,
        "vision_tokens": int(vision_idx.numel()),
        "llm_grid": {"t": t, "h": gh, "w": gw},
        "last_layers": args.last_layers,
        "overlay": str(overlay_path),
        "panel": str(panel_path),
        "keyword_maps": keyword_meta,
    }
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] overlay: {overlay_path}")
    print(f"[ok] panel:   {panel_path}")
    print(f"[ok] json:    {json_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
