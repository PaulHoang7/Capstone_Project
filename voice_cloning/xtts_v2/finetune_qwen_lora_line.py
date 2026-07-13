"""LoRA fine-tune Qwen2.5-VL-7B-Instruct on Vietnamese comic OCR — LINE LEVEL.

Differences vs finetune_qwen_lora.py (bubble-level baseline):
    - Input: per-line crops produced by prepare_line_pairs_vn.py (line_pairs_vn.jsonl)
    - Target text: single line (no \\n inside response)
    - Smaller max_pixels per sample (lines are ~30-60px tall)
    - Shorter max_new_tokens at eval (lines are <80 chars)
    - System prompt updated: "this is ONE line, return the text on it"

Anti-hallucination guards reused from baseline:
    - Return empty if illegible
    - Loss masked to response tokens only
    - No prompt echo

GPU budget: 24 GB → Qwen2.5-VL-7B in bf16 + LoRA fits without QLoRA.
    Pass --qlora-4bit to enable 4-bit quantization for tighter fit / larger batch.

Usage:
    conda run -n comic_ocr python finetune_qwen_lora_line.py \\
        --jsonl /home/bes/Desktop/Tin/labeling_task_v4/line_pairs_vn.jsonl \\
        --output-dir /mnt/nfs-data/tin_dataset/checkpoints/qwen25vl_7b_vncomic_line_lora_v1 \\
        --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --epochs 3 --lr 1e-4 --batch-size 1 --grad-accum 8
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


SYSTEM_PROMPT = """Bạn là OCR engine đọc MỘT DÒNG văn bản tiếng Việt từ truyện tranh.

QUY TẮC TUYỆT ĐỐI:
1. CHỈ trả về văn bản trên 1 dòng được cung cấp. KHÔNG giải thích, KHÔNG mô tả.
2. Nếu không đọc được → trả về CHUỖI RỖNG.
3. KHÔNG xuống dòng. KHÔNG echo prompt. KHÔNG nói "Tôi không thể...".
4. Giữ chính xác dấu thanh điệu (sắc/huyền/hỏi/ngã/nặng) và dấu chữ (ă â đ ê ô ơ ư).
5. Giữ đúng case (chữ HOA hay chữ thường) như xuất hiện trong ảnh."""


USER_PROMPT = "Đọc văn bản trên dòng này. Trả về chỉ văn bản, không xuống dòng."


class LineOCRDataset(Dataset):
    """Loads (line image, line text) pairs from JSONL."""

    def __init__(self, records, processor, max_pixels=384 * 384):
        self.records = records
        self.processor = processor
        self.max_pixels = max_pixels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        gt = (r.get("text") or "").strip()
        # Strip any stray newlines (should already be single-line)
        gt = gt.replace("\n", " ").replace("\r", " ").strip()

        image = Image.open(r["image"]).convert("RGB")
        w, h = image.size
        if w * h > self.max_pixels:
            scale = (self.max_pixels / (w * h)) ** 0.5
            image = image.resize(
                (max(8, int(w * scale)), max(8, int(h * scale))), Image.LANCZOS
            )
        return {"image": image, "gt": gt, "id": f"{r.get('source_id','?')}_L{r.get('line_idx',0)}"}


def collate_fn(batch, processor):
    from qwen_vl_utils import process_vision_info

    texts = []
    messages_with_imgs = []
    for b in batch:
        gt = b["gt"]
        msg_template = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": USER_PROMPT},
            ]},
            {"role": "assistant", "content": gt},
        ]
        texts.append(processor.apply_chat_template(msg_template, tokenize=False))

        messages_with_imgs.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": b["image"]},
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

    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    asst_token_str = "<|im_start|>assistant\n"
    asst_ids = processor.tokenizer(asst_token_str, add_special_tokens=False)["input_ids"]
    pad_id = processor.tokenizer.pad_token_id

    for i in range(input_ids.shape[0]):
        ids = input_ids[i].tolist()
        asst_start = -1
        for j in range(len(ids) - len(asst_ids)):
            if ids[j:j + len(asst_ids)] == asst_ids:
                asst_start = j + len(asst_ids)
                break
        if asst_start > 0:
            labels[i, :asst_start] = -100
        if pad_id is not None:
            labels[i][input_ids[i] == pad_id] = -100

    inputs["labels"] = labels
    return inputs


def cer(pred, gt):
    pred = (pred or "").strip()
    gt = (gt or "").strip()
    if not gt:
        return 0.0 if not pred else 1.0
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
def evaluate(model, processor, valid_records, max_eval=200):
    from qwen_vl_utils import process_vision_info

    model.eval()
    cers, upper_cers, lower_cers = [], [], []
    n_halluc = n_long = 0

    for r in valid_records[:max_eval]:
        try:
            image = Image.open(r["image"]).convert("RGB")
        except Exception:
            continue
        gt = (r.get("text") or "").strip().replace("\n", " ")

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
        out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
        pred = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        c = cer(pred, gt)
        cers.append(c)
        if gt and gt == gt.upper() and any(ch.isalpha() for ch in gt):
            upper_cers.append(c)
        elif gt:
            lower_cers.append(c)

        if "tôi không thể" in pred.lower() or "đây là một" in pred.lower():
            n_halluc += 1
        if len(pred) > max(len(gt) * 5, 80):
            n_long += 1

    model.train()
    if not cers:
        return {"n": 0, "mean_cer": 0, "upper_cer": 0, "lower_cer": 0, "halluc": 0, "long": 0}
    return {
        "n": len(cers),
        "mean_cer": sum(cers) / len(cers),
        "upper_cer": (sum(upper_cers) / len(upper_cers)) if upper_cers else None,
        "n_upper": len(upper_cers),
        "lower_cer": (sum(lower_cers) / len(lower_cers)) if lower_cers else None,
        "n_lower": len(lower_cers),
        "halluc": n_halluc,
        "long": n_long,
    }


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="line_pairs_vn.jsonl from prepare_line_pairs_vn.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-pixels", type=int, default=384 * 384,
                   help="Cap pixels per line (smaller than bubble-level)")
    p.add_argument("--valid-pct", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--qlora-4bit", action="store_true",
                   help="Use 4-bit base + LoRA (QLoRA) — saves ~7GB, slower ~30%")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    records = load_jsonl(args.jsonl)
    print(f"Loaded {len(records)} line pairs from {args.jsonl}")

    # Stratify split by series so val covers all series
    by_series = {}
    for r in records:
        by_series.setdefault(r.get("series", "?"), []).append(r)
    train_recs, valid_recs = [], []
    for s, rs in by_series.items():
        random.shuffle(rs)
        n_valid = max(1, int(len(rs) * args.valid_pct))
        valid_recs.extend(rs[:n_valid])
        train_recs.extend(rs[n_valid:])
    random.shuffle(train_recs)
    print(f"Train: {len(train_recs)}, Valid: {len(valid_recs)}")
    with open(out_dir / "data_split.json", "w") as f:
        json.dump({
            "train": [f"{r.get('source_id','?')}_L{r.get('line_idx',0)}" for r in train_recs],
            "valid": [f"{r.get('source_id','?')}_L{r.get('line_idx',0)}" for r in valid_recs],
        }, f, indent=2)

    print(f"\nLoading {args.model_id}...")
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    processor = AutoProcessor.from_pretrained(args.model_id)

    load_kwargs = {"device_map": "cuda:0"}
    if args.qlora_4bit:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_cfg
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_id, **load_kwargs)
    if args.qlora_4bit:
        model = prepare_model_for_kbit_training(model)
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.train()

    train_ds = LineOCRDataset(train_recs, processor, max_pixels=args.max_pixels)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, num_workers=2,
        collate_fn=lambda b: collate_fn(b, processor),
    )
    total_steps = max(1, (len(train_loader) // args.grad_accum) * args.epochs)
    print(f"Total optimizer steps: {total_steps}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
                    s = evaluate(model, processor, valid_recs, max_eval=100)
                    print(f"  EVAL  n={s['n']}  CER={s['mean_cer']:.4f}  "
                          f"upper={s['upper_cer']}({s['n_upper']})  "
                          f"lower={s['lower_cer']}({s['n_lower']})  "
                          f"halluc={s['halluc']}  long={s['long']}")
                    if s["mean_cer"] < best_cer:
                        best_cer = s["mean_cer"]
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

    print("\n=== Final evaluation ===")
    final = evaluate(model, processor, valid_recs, max_eval=400)
    print(f"FINAL  n={final['n']}  CER={final['mean_cer']:.4f}")
    print(f"  upper CER: {final['upper_cer']} on {final['n_upper']} lines")
    print(f"  lower CER: {final['lower_cer']} on {final['n_lower']} lines")
    print(f"  hallucination: {final['halluc']}  too-long: {final['long']}")

    model.save_pretrained(out_dir / "final")
    with open(out_dir / "final_stats.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
