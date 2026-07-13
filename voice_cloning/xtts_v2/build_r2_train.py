"""Build a round-2 targeted train set from v2's ACTUAL errors (not prefill's).

Joins v2 predictions (from eval_ocr_bubble.py --out-csv on the train set) back to the
full train CSV (for image/series/page), recomputes error_type from v2-pred-vs-GT, and:
  - duplicates v2-ERROR rows x2  (concentrate gradient where v2 is actually wrong)
  - keeps ALL v2-CORRECT rows x1 (regularizer to prevent the `correct` regression seen in round 1)
Note: the trainer ignores any `weight` column, so upweighting MUST be physical duplication.

Usage:
    python build_r2_train.py --pred train_v2_pred.csv --full ocr_ft_train.csv --out ocr_ft_train_r2.csv
"""
import argparse
import csv
import re
import unicodedata


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", (s or "").strip())).upper()


def strip_dia(s):
    s = unicodedata.normalize("NFD", s.upper())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("Đ", "D")  # Đ -> D


def words(s):
    return re.findall(r"[A-Z]+", strip_dia(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="v2 predictions CSV (id,gt,pred from eval_ocr_bubble)")
    ap.add_argument("--full", required=True, help="full train CSV with image/series/page columns")
    ap.add_argument("--out", required=True)
    ap.add_argument("--err-dup", type=int, default=2, help="how many copies of each v2-error row")
    args = ap.parse_args()

    pred = {}
    with open(args.pred, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            pred[r["id"]] = r.get("pred", "")

    full = {}
    cols_order = None
    with open(args.full, encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols_order = rd.fieldnames
        for r in rd:
            full[r["id"]] = r

    out_rows = []
    n_err = n_cor = n_missing = 0
    for rid, row in full.items():
        if rid not in pred:
            n_missing += 1
            continue
        gt = (row.get("corrected_text") or "").strip()
        pv = pred[rid]
        is_err = norm(pv) != norm(gt)
        if is_err:
            et = "diacritic" if words(pv) == words(gt) else "structural"
            n_err += 1
            copies = args.err_dup
        else:
            et = "correct"
            n_cor += 1
            copies = 1
        out = {k: row.get(k, "") for k in ["id", "series", "page", "order", "image", "prefill", "corrected_text"]}
        out["error_type"] = et
        out["skip"] = ""
        out_rows.extend([out] * copies)

    cols = ["id", "series", "page", "order", "image", "prefill", "corrected_text", "error_type", "skip"]
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    total = n_err + n_cor
    print(f"v2 on train: {total} rows | ERROR {n_err} ({n_err/max(total,1)*100:.1f}%)  CORRECT {n_cor}  missing-pred {n_missing}")
    print(f"Round-2 train written: {args.out}")
    print(f"  rows = {n_err}x{args.err_dup} errors + {n_cor} correct = {len(out_rows)} total "
          f"(error-instances {n_err*args.err_dup}/{len(out_rows)} = {n_err*args.err_dup/max(len(out_rows),1)*100:.0f}%)")


if __name__ == "__main__":
    main()
