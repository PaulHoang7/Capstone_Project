"""Stage 1: Run YOLO + Qwen-VL on comic pages, dump bubble JSON per page.

Runs in `comic_ocr` conda env. Output JSONs feed into stage 2 (XTTS synthesis).
"""
import argparse, json, sys, time, types
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pages-dir", required=True)
    p.add_argument("--out", required=True, help="Output dir for per-page JSON")
    p.add_argument("--max-pages", type=int, default=5)
    p.add_argument("--skip-pages", type=int, default=0, help="Skip N first pages")
    p.add_argument("--yolo-ckpt", default="/mnt/nfs-data/tin_dataset/checkpoints/yolo_comic.pt")
    p.add_argument("--speaker-mlp", default="/mnt/nfs-data/tin_dataset/comic/speaker_attribution/speaker_mlp.pt")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, "/home/bes/Desktop/Tin")
    cv_args = types.SimpleNamespace(
        weights=args.yolo_ckpt,
        speaker_model=args.speaker_mlp,
        speaker_scaler="/mnt/nfs-data/tin_dataset/comic/speaker_attribution/speaker_scaler.pkl",
        ocr_engine="qwen-vl",
        lang="vi",
        direction="ltr",
        no_ocr=False,
        no_face=True,
        no_gpu=True,
        paddle_only=False,
        char_db=None,
        verbose=False,
        yolo_conf=0.25,
        rule_based=False,
    )

    print("[1/3] Initializing CV pipeline (YOLO + Qwen-VL)...")
    from Capstone_project.scripts.comic_pipeline import init_pipeline, process_page
    cv_models = init_pipeline(cv_args)
    print("  Ready")

    pages_dir = Path(args.pages_dir)
    all_pages = sorted(list(pages_dir.glob("*.jpg")) +
                       list(pages_dir.glob("*.webp")) +
                       list(pages_dir.glob("*.png")))
    page_files = all_pages[args.skip_pages : args.skip_pages + args.max_pages]
    print(f"[2/3] {len(page_files)} pages to process")

    all_results = []
    for page_idx, img_path in enumerate(page_files):
        page_name = img_path.stem
        print(f"\n[{page_idx+1}/{len(page_files)}] {page_name}")
        t0 = time.time()
        try:
            result = process_page(
                img_path,
                yolo_model=cv_models["yolo"],
                ocr_pipeline=cv_models["ocr"],
                face_extractor=cv_models["extractor"],
                char_db=cv_models["char_db"],
                speaker_model_path=cv_models["speaker_model"],
                speaker_scaler_path=cv_models["speaker_scaler"],
                direction=cv_args.direction,
                rule_based=cv_args.rule_based,
                use_gpu=not cv_args.no_gpu,
                yolo_conf=cv_args.yolo_conf,
            )
        except Exception as exc:
            print(f"  CV failed: {exc}")
            continue

        n_bubbles = len(result.get("bubbles", []))
        print(f"  {n_bubbles} bubbles in {time.time()-t0:.1f}s")

        # Dump JSON
        json_path = out_dir / f"{page_idx+1:03d}_{page_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        all_results.append({"page_idx": page_idx+1, "name": page_name,
                            "json": str(json_path), "n_bubbles": n_bubbles})

    # Write index
    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n[3/3] Done. {len(all_results)} pages → {out_dir}")


if __name__ == "__main__":
    main()
