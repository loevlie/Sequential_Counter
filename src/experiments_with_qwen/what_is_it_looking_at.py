import os

# Optional: allocator hint (harmless if ignored)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import torch
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from PIL import Image, ImageFilter
from matplotlib import pyplot as plt

# -------------------------
# 0. Device / model / processor
# -------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen3-VL-2B-Instruct"
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map={"": DEVICE},
    attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(model_name)

# Reduce visual token budget to save VRAM
# (Qwen3-VL manual suggests controlling tokens via size)
processor.image_processor.size = {"shortest_edge": 256, "longest_edge": 256}

model.config.use_cache = False
model.eval()


# -------------------------
# 1. Basic helpers
# -------------------------

def make_blurred_image(image: Image.Image, radius: int = 32) -> Image.Image:
    """Blur helper (not strictly needed for occlusion, but handy)."""
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def build_inputs(image: Image.Image, question: str):
    """
    Build a FULL inputs dict, including image info, for Qwen3-VL.

    Returns:
        inputs: dict of tensors on DEVICE (input_ids, pixel_values, image_grid_thw, ...)
        prompt_len: length of the prompt tokens (before generation)
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Sometimes token_type_ids is present but unused
    inputs.pop("token_type_ids", None)

    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]
    return inputs, prompt_len


@torch.no_grad()
def compute_token_logprobs_from_ids(model, base_inputs, generated_ids):
    """
    Compute per-token log probs, reusing the
    vision + text inputs in `base_inputs`.

    Args:
        model: Qwen3VLForConditionalGeneration
        base_inputs: dict from build_inputs (contains image + prompt info)
        generated_ids: full sequence (prompt + answer) from model.generate

    Returns:
        token_logp: (B, T-1) log p(token_{t+1})
        labels_shift: (B, T-1) the predicted token ids
    """
    full_inputs = {k: v for k, v in base_inputs.items()}
    full_inputs["input_ids"] = generated_ids.to(DEVICE)
    full_inputs["attention_mask"] = torch.ones_like(
        generated_ids, device=DEVICE
    )

    outputs = model(**full_inputs)
    logits = outputs.logits.float()  # (B, T, V)

    # Standard LM shifting: logits at position t predict token at t+1
    logits_shift = logits[:, :-1, :]        # (B, T-1, V)
    labels_shift = generated_ids[:, 1:]     # (B, T-1)

    log_probs = F.log_softmax(logits_shift, dim=-1)  # (B, T-1, V)

    token_logp = log_probs.gather(
        dim=-1, index=labels_shift.unsqueeze(-1)
    ).squeeze(-1)  # (B, T-1)

    return token_logp, labels_shift


def compute_answer_score(model, base_inputs, generated_ids, prompt_len):
    """
    Compute scalar answer score = sum log p(answer tokens | image, question).

    We treat "answer tokens" as positions after the prompt.
    """
    token_logp, labels_shift = compute_token_logprobs_from_ids(
        model, base_inputs, generated_ids
    )
    # token_logp: (1, T-1)
    ll = token_logp[0]  # (T-1,)

    T_minus1 = ll.shape[0]
    pos = torch.arange(T_minus1, device=ll.device)

    # prompt tokens end at index (prompt_len - 1) in labels_shift space
    answer_mask = pos >= (prompt_len - 1)

    if answer_mask.sum() == 0:
        # degenerate edge case: no answer tokens, just sum everything
        score = ll.sum()
    else:
        score = ll[answer_mask].sum()

    return score, token_logp, labels_shift


def occlude_cell(image: Image.Image, i: int, j: int, H_cells: int, W_cells: int,
                 mode: str = "gray") -> Image.Image:
    """
    Occlude one cell in an H_cells x W_cells grid over the image.
    mode = "gray" or "blur"
    """
    img = image.copy()
    W, H = img.size
    cell_w = W // W_cells
    cell_h = H // H_cells

    left = j * cell_w
    upper = i * cell_h
    right = (j + 1) * cell_w if j < W_cells - 1 else W
    lower = (i + 1) * cell_h if i < H_cells - 1 else H

    if mode == "gray":
        patch = Image.new("RGB", (right - left, lower - upper), (127, 127, 127))
    else:  # mode == "blur"
        region = img.crop((left, upper, right, lower))
        patch = region.filter(ImageFilter.GaussianBlur(radius=16))

    img.paste(patch, (left, upper))
    return img


# -------------------------
# 2. Main occlusion heatmap
# -------------------------

def compute_heatmap_qwen3_vl_occlusion(
    image: Image.Image,
    question: str,
    grid_size=(10, 10),
    max_new_tokens: int = 8,
    occlusion_mode: str = "gray",
):
    """
    Occlusion-based heatmap:
    - Generate answer on original image
    - Compute answer logprob
    - For each spatial cell, occlude that region and recompute answer logprob
    - Importance = drop in logprob when that region is occluded
    """
    H_cells, W_cells = grid_size

    # 1. Real-image inputs and answer
    inputs_real, prompt_len = build_inputs(image, question)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs_real,
            max_new_tokens=max_new_tokens,
        )

    # Decode answer text for convenience
    answer_only_ids = [out[prompt_len:] for out in generated_ids]
    answer_text = processor.batch_decode(
        answer_only_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # 2. Baseline answer score (real image)
    base_score, base_logp, base_labels = compute_answer_score(
        model, inputs_real, generated_ids, prompt_len
    )

    # 3. For each spatial cell: occlude and measure score drop
    importance = torch.zeros(H_cells, W_cells, device=DEVICE, dtype=torch.float32)

    for i in range(H_cells):
        for j in range(W_cells):
            occ_img = occlude_cell(image, i, j, H_cells, W_cells, mode=occlusion_mode)

            inputs_occ, prompt_len_occ = build_inputs(occ_img, question)
            # Sanity: prompt should be same length
            # (we don't strictly rely on it though)
            assert prompt_len_occ == prompt_len, "Prompt length changed – unexpected."

            score_occ, logp_occ, labels_occ = compute_answer_score(
                model, inputs_occ, generated_ids, prompt_len
            )

            # labels should match (we're scoring the same generated_ids)
            assert torch.all(labels_occ == base_labels)

            # importance of this region = how much score drops when it's occluded
            diff = (base_score - score_occ).clamp(min=0.0)
            importance[i, j] = diff

            torch.cuda.empty_cache()

    # 4. Normalize importance to [0,1]
    if importance.max() > 0:
        importance = importance / importance.max()

    # 5. Upsample to image resolution
    H_img, W_img = image.size[1], image.size[0]
    heatmap_small = importance.view(1, 1, H_cells, W_cells)
    heatmap = F.interpolate(
        heatmap_small,
        size=(H_img, W_img),
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu()

    # Safety re-normalization
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap, answer_text


# -------------------------
# 3. Demo
# -------------------------

if __name__ == "__main__":
    img = Image.open("/media/M2SSD/FSC147/images_384_VarV2/194.jpg").convert("RGB")
    question = "How many peaches are in the image?  Answer with a single number."

    heatmap, answer_text = compute_heatmap_qwen3_vl_occlusion(
        img,
        question,
        grid_size=(8, 8),       # coarser = faster, finer = more precise
        max_new_tokens=24,
        occlusion_mode="gray",  # or "blur"
    )

    print("Answer:", answer_text)

    # Visualize overlay
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(heatmap.numpy(), alpha=0.5)
    plt.axis("off")
    plt.show()
