"""LoRA fine-tune Qwen2.5-VL-3B-Instruct on Vietnamese manga OCR.

Goals:
    1. Reduce CER from ~0.15 → ~0.05 on bubble OCR
    2. Eliminate hallucination (Qwen returning prompt-echo text instead of bubble content)

Anti-hallucination strategy:
    - System prompt emphasizes "return EMPTY if can't read"
    - Use skip='x' rows as explicit "return empty string" examples
    - Loss masked to RESPONSE only (model learns to output text, not echo prompt)

Data:
    /home/bes/Desktop/Tin/labels_edited.csv (5695 labeled + 9 skip = 5704 train rows)
    /home/bes/Desktop/Tin/labeling_task_v4/bubble_crops/*.png

LoRA config:
    - Rank 16, alpha 32, dropout 0.05
    - Target: q/k/v/o projections of language model
    - Frozen: vision encoder + base LM

Usage:
    conda run -n comic_ocr python finetune_qwen_lora.py \\
        --epochs 3 \\
        --lr 1e-4 \\
        --batch-size 1 \\
        --grad-accum 8 \\
        --output-dir /mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_3b_vncomic_lora_v1
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image


SYSTEM_PROMPT = """Bạn là OCR engine đọc bong bóng thoại tiếng Việt trong truyện tranh.

QUY TẮC TUYỆT ĐỐI:
1. CHỈ trả về văn bản đọc được trong bong bóng. KHÔNG giải thích, KHÔNG mô tả.
2. Nếu không đọc được rõ HOẶC bubble không có chữ → trả về CHUỖI RỖNG (không gõ gì).
3. KHÔNG bịa thêm text. KHÔNG echo prompt. KHÔNG nói "Tôi không thể...".
4. Giữ chính xác dấu thanh điệu (sắc/huyền/hỏi/ngã/nặng) và dấu chữ (ă â đ ê ô ơ ư).
5. Output viết chữ thường (lowercase), trừ tên riêng và đầu câu."""


USER_PROMPT = "Đọc văn bản trong bong bóng thoại này. Trả về chỉ văn bản đọc được, hoặc chuỗi rỗng nếu không có."


class BubbleOCRDataset(Dataset):
    """Loads (crop image, ground truth text) pairs from CSV."""

    def __init__(self, rows, crops_dir, processor, max_pixels=512*512):
        self.rows = rows
        self.crops_dir = Path(crops_dir)
        self.processor = processor
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img_str = r["image"]
        if img_str.startswith("/"):
            img_path = Path(img_str)
        else:
            img_path = self.crops_dir / img_str.split("/")[-1]
        # ground truth: corrected_text if labeled, else empty (skip='x' case)
        gt = r.get("corrected_text", "").strip()
        if r.get("skip", "").strip() == "x":
            gt = ""  # explicit empty for skip rows

        image = Image.open(img_path).convert("RGB")
        # Resize if too big to control memory
        w, h = image.size
        if w * h > self.max_pixels:
            scale = (self.max_pixels / (w * h)) ** 0.5
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        return {"image": image, "gt": gt, "id": r["id"]}


def collate_fn(batch, processor):
    """Build messages format and tokenize. Loss mask: only response tokens contribute to loss."""
    from qwen_vl_utils import process_vision_info

    images = [b["image"] for b in batch]
    gts = [b["gt"] for b in batch]

    texts = []
    for gt in gts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": gt},
        ]
        text = processor.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    # Process_vision_info expects messages — build a parallel list with actual images
    messages_with_imgs = []
    for img, gt in zip(images, gts):
        messages_with_imgs.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": gt},
        ])
    img_inputs, vid_inputs = process_vision_info(messages_with_imgs[0])
    if len(messages_with_imgs) > 1:
        for m in messages_with_imgs[1:]:
            ii, vi = process_vision_info(m)
            img_inputs += ii or []
            vid_inputs = (vid_inputs or []) + (vi or [])

    inputs = processor(
        text=texts, images=img_inputs, videos=vid_inputs,
        padding=True, return_tensors="pt",
    )

    # Mask labels: only train on response (assistant) tokens
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    # Find assistant marker: '<|im_start|>assistant\n'
    asst_token_str = "<|im_start|>assistant\n"
    asst_ids = processor.tokenizer(asst_token_str, add_special_tokens=False)["input_ids"]
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id

    for i in range(input_ids.shape[0]):
        ids = input_ids[i].tolist()
        # Find where assistant response starts
        asst_start = -1
        for j in range(len(ids) - len(asst_ids)):
            if ids[j:j + len(asst_ids)] == asst_ids:
                asst_start = j + len(asst_ids)
                break
        # Mask everything before assistant response
        if asst_start > 0:
            labels[i, :asst_start] = -100
        # Mask padding
        labels[i][input_ids[i] == pad_id] = -100

    inputs["labels"] = labels
    return inputs


def cer(pred, gt):
    pred = pred.strip()
    gt = gt.strip()
    if not gt:
        return 0.0 if not pred else 1.0
    # Levenshtein
    s1, s2 = pred, gt
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1) / max(len(gt), 1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        cur = [i + 1]
        for j, c2 in enumerate(s2):
            cur.append(min(prev[j + 1] + 1, cur[j] + 1, prev[j] + (c1 != c2)))
        prev = cur
    return prev[-1] / len(gt)


@torch.no_grad()
def evaluate(model, processor, valid_rows, crops_dir, max_eval=200):
    """Quick eval: predict + measure CER + hallucination rate."""
    from qwen_vl_utils import process_vision_info

    model.eval()
    crops_dir = Path(crops_dir)
    cers = []
    n_halluc = 0
    n_long = 0

    for r in valid_rows[:max_eval]:
        img_path = crops_dir / r["image"].split("/")[-1]
        if not img_path.exists():
            continue
        gt = r.get("corrected_text", "").strip()
        if r.get("skip", "").strip() == "x":
            gt = ""

        image = Image.open(img_path).convert("RGB")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inp, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=img_inp, padding=True, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        pred = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        cer_val = cer(pred.lower(), gt.lower())
        cers.append(cer_val)
        # Quick hallucination check
        if "tôi không thể" in pred.lower() or "đây là một" in pred.lower():
            n_halluc += 1
        if len(pred) > max(len(gt) * 5, 200):
            n_long += 1

    model.train()
    if not cers:
        return {"n": 0, "mean_cer": 0, "halluc": 0, "long": 0}
    return {
        "n": len(cers),
        "mean_cer": sum(cers) / len(cers),
        "halluc": n_halluc,
        "long": n_long,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/home/bes/Desktop/Tin/labels_edited.csv")
    p.add_argument("--crops-dir", default="/home/bes/Desktop/Tin/labeling_task_v4/bubble_crops")
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output-dir", default="/mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_3b_vncomic_lora_v1")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-target", default="attn",
                   choices=["attn", "attn+mlp"],
                   help="attn = q/k/v/o only (default); attn+mlp adds gate/up/down_proj")
    p.add_argument("--resume-adapter", default=None,
                   help="Warm-start: continue training from an existing LoRA adapter dir "
                        "(e.g. .../qwen25vl_7b_vncomic_uppercase_lora_v2/best). When set, "
                        "--lora-rank/--lora-alpha/--lora-target are ignored (adapter config is reused).")
    p.add_argument("--max-pixels", type=int, default=512 * 512)
    p.add_argument("--valid-pct", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0,
                   help="Stop after N optimizer steps (0=no limit). Used for sanity check.")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # Load CSV
    with open(args.csv, encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    # Keep rows with corrected_text OR skip='x' (the latter = "return empty")
    rows = [r for r in all_rows if r.get("corrected_text", "").strip() or r.get("skip", "").strip() == "x"]
    print(f"Total usable rows: {len(rows)} (labeled={sum(1 for r in rows if r.get('corrected_text','').strip())}, skip={sum(1 for r in rows if r.get('skip','').strip()=='x')})")

    # Stratified split by series
    by_series = {}
    for r in rows:
        by_series.setdefault(r.get("series", "?"), []).append(r)
    train_rows, valid_rows = [], []
    for s, rs in by_series.items():
        random.shuffle(rs)
        n_valid = max(1, int(len(rs) * args.valid_pct))
        valid_rows.extend(rs[:n_valid])
        train_rows.extend(rs[n_valid:])
    random.shuffle(train_rows)
    print(f"Train: {len(train_rows)}, Valid: {len(valid_rows)}")

    # Save split
    with open(out_dir / "data_split.json", "w") as f:
        json.dump({
            "train_ids": [r["id"] for r in train_rows],
            "valid_ids": [r["id"] for r in valid_rows],
        }, f, indent=2)

    # Load model + processor
    print(f"\nLoading {args.model_id}...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="cuda:0"
    )
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Apply LoRA — language model only (vision frozen)
    if args.resume_adapter:
        # Warm-start: continue training from an existing adapter (reuses its rank/alpha/targets)
        print(f"Warm-starting (continue training) from adapter: {args.resume_adapter}")
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if args.lora_target == "attn+mlp":
            target_modules += ["gate_proj", "up_proj", "down_proj"]
        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Required so gradients flow through grad-checkpointed blocks (esp. the resume/warm-start
    # path where get_peft_model's auto-hook isn't applied) — else loss.requires_grad is False.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()

    # Datasets
    train_ds = BubbleOCRDataset(train_rows, args.crops_dir, processor, max_pixels=args.max_pixels)
    valid_ds_rows = valid_rows  # use rows directly for eval

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=2,
        collate_fn=lambda b: collate_fn(b, processor),
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    print(f"Total optimizer steps: {total_steps}")

    # Optimizer + scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    log_every = 25
    global_step = 0
    accum_count = 0
    t0 = time.time()
    best_cer = float("inf")

    print("\n=== Starting training ===")
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to("cuda:0") for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            accum_count += 1

            if accum_count == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                if global_step % log_every == 0:
                    elapsed = time.time() - t0
                    eta_min = elapsed / max(global_step, 1) * (total_steps - global_step) / 60
                    cur_lr = scheduler.get_last_lr()[0]
                    print(f"  ep{epoch+1}/{args.epochs}  step {global_step}/{total_steps}  "
                          f"loss={outputs.loss.item():.3f}  lr={cur_lr:.2e}  "
                          f"elapsed={elapsed/60:.1f}m  eta={eta_min:.1f}m")

                if global_step % args.eval_every == 0:
                    print(f"  >>> Eval at step {global_step}...")
                    eval_stats = evaluate(model, processor, valid_ds_rows, args.crops_dir, max_eval=100)
                    print(f"  EVAL  n={eval_stats['n']}  CER={eval_stats['mean_cer']:.4f}  "
                          f"halluc={eval_stats['halluc']}  long={eval_stats['long']}")
                    if eval_stats["mean_cer"] < best_cer:
                        best_cer = eval_stats["mean_cer"]
                        model.save_pretrained(out_dir / "best")
                        print(f"  BEST checkpoint saved (CER={best_cer:.4f})")

                if global_step % args.save_every == 0:
                    model.save_pretrained(out_dir / f"step_{global_step}")
                    print(f"  Checkpoint saved at step {global_step}")

                if args.max_steps > 0 and global_step >= args.max_steps:
                    print(f"  Reached --max-steps={args.max_steps}, stopping early.")
                    break
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Final eval + save
    print("\n=== Final evaluation ===")
    final_stats = evaluate(model, processor, valid_ds_rows, args.crops_dir, max_eval=200)
    print(f"FINAL CER on {final_stats['n']} valid rows: {final_stats['mean_cer']:.4f}")
    print(f"Hallucination cases: {final_stats['halluc']}")
    print(f"Too-long cases: {final_stats['long']}")

    model.save_pretrained(out_dir / "final")
    with open(out_dir / "final_stats.json", "w") as f:
        json.dump(final_stats, f, indent=2)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
