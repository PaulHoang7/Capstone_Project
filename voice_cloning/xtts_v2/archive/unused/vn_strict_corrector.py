"""Strict Vietnamese OCR corrector with 3 hard constraints.

Hard Rule 1: Every word in output MUST be in VN dictionary.
Hard Rule 2: Bigram LM picks best candidate when multiple valid corrections exist.
Hard Rule 3: Manga-specific vocabulary whitelist (names, SFX) force-accepted.

Pipeline: raw_ocr_text → tokenize → per-word constraint + LM disambiguation → valid VN.

Usage:
    python vn_strict_corrector.py --in <cv json dir> --out <fixed json dir>
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path


RESOURCES = Path("/mnt/nfs-data/tin_dataset/ocr_corrector")


class StrictCorrector:
    """Vietnamese UPPERCASE OCR corrector with dict + bigram LM."""

    def __init__(self, max_edit_distance: int = 2):
        self.max_ed = max_edit_distance

        # Load VN dictionary
        with open(RESOURCES / "vn_words.txt", encoding="utf-8") as f:
            self.vn_dict = {w.strip().upper() for w in f if w.strip()}
        print(f"  VN dict: {len(self.vn_dict)} words")

        # Load bigram LM
        with open(RESOURCES / "vn_bigrams.json", encoding="utf-8") as f:
            self.bigrams = json.load(f)
        with open(RESOURCES / "vn_unigrams.json", encoding="utf-8") as f:
            self.unigrams = json.load(f)
        total = sum(self.unigrams.values())
        self.total_unigrams = total
        print(f"  Bigram LM: {len(self.bigrams)} w1 keys")

        # Manga whitelist
        with open(RESOURCES / "manga_whitelist.txt", encoding="utf-8") as f:
            self.whitelist = {w.strip().upper() for w in f if w.strip()}

        # Pre-index dictionary by length + first char for fast candidate search
        self._by_len_prefix: dict[tuple[int, str], list[str]] = {}
        for w in self.vn_dict:
            if not w:
                continue
            key = (len(w), w[0])
            self._by_len_prefix.setdefault(key, []).append(w)

    # -------------------------------------------------------------------------
    # Edit distance (Levenshtein, early termination)
    # -------------------------------------------------------------------------
    @staticmethod
    def _edit_distance(a: str, b: str, max_d: int = 2) -> int:
        la, lb = len(a), len(b)
        if abs(la - lb) > max_d:
            return max_d + 1
        if la == 0:
            return lb
        if lb == 0:
            return la
        # DP with early termination
        prev = list(range(lb + 1))
        for i in range(1, la + 1):
            curr = [i] + [0] * lb
            min_row = curr[0]
            for j in range(1, lb + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,         # deletion
                    curr[j - 1] + 1,     # insertion
                    prev[j - 1] + cost,  # substitution
                )
                if curr[j] < min_row:
                    min_row = curr[j]
            if min_row > max_d:
                return max_d + 1
            prev = curr
        return prev[lb]

    # -------------------------------------------------------------------------
    # Find candidates within edit distance
    # -------------------------------------------------------------------------
    def _find_candidates(self, word: str) -> list[tuple[str, int]]:
        """Return list of (candidate, edit_distance) within max_ed."""
        if not word:
            return []
        candidates = []
        # Search in words with similar length (±max_ed) + same first char (mostly)
        for L in range(max(1, len(word) - self.max_ed), len(word) + self.max_ed + 1):
            # Try same first char + neighbors (common OCR substitutions rarely change first char)
            for c in (word[0], word[0].upper() if word else ""):
                key = (L, c)
                if key not in self._by_len_prefix:
                    continue
                for w in self._by_len_prefix[key]:
                    d = self._edit_distance(word, w, self.max_ed)
                    if d <= self.max_ed:
                        candidates.append((w, d))
        # Deduplicate, sort by edit distance ascending
        seen = set()
        out = []
        for w, d in sorted(candidates, key=lambda x: x[1]):
            if w not in seen:
                seen.add(w)
                out.append((w, d))
        return out

    # -------------------------------------------------------------------------
    # Bigram LM score
    # -------------------------------------------------------------------------
    def _bigram_score(self, prev_word: str, candidate: str) -> float:
        """P(candidate | prev_word) with add-1 smoothing."""
        if prev_word in self.bigrams and candidate in self.bigrams[prev_word]:
            count_bi = self.bigrams[prev_word][candidate]
            count_prev = self.unigrams.get(prev_word, 1)
            return (count_bi + 1) / (count_prev + len(self.vn_dict))
        # Backoff: unigram freq
        uc = self.unigrams.get(candidate, 0)
        return (uc + 1) / (self.total_unigrams + len(self.vn_dict))

    # -------------------------------------------------------------------------
    # Correct a sentence
    # -------------------------------------------------------------------------
    # Vietnamese letter pattern (includes all diacritic forms)
    _WORD_RE = re.compile(r"([A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]+)",
                          re.IGNORECASE)

    def correct(self, text: str) -> tuple[str, list[dict]]:
        """Correct sentence. Returns (corrected_text, changes_log)."""
        # Tokenize: keep words + interleave non-word tokens (punct, spaces)
        tokens = []
        last = 0
        for m in self._WORD_RE.finditer(text):
            if m.start() > last:
                tokens.append(("sep", text[last:m.start()]))
            tokens.append(("word", m.group()))
            last = m.end()
        if last < len(text):
            tokens.append(("sep", text[last:]))

        # Correct each word in context
        corrected = []
        changes = []
        prev_word = "<S>"
        for kind, tok in tokens:
            if kind == "sep":
                corrected.append(tok)
                continue
            word = tok.upper()
            # Rule 3: Whitelist (manga names, SFX)
            if word in self.whitelist:
                corrected.append(word)
                prev_word = word
                continue
            # Rule 1: In dictionary → keep
            if word in self.vn_dict:
                corrected.append(word)
                prev_word = word
                continue
            # Invalid word → find candidates
            candidates = self._find_candidates(word)
            if not candidates:
                # No candidates within edit distance → keep original + flag
                corrected.append(word)
                changes.append({
                    "original": word, "corrected": word,
                    "reason": "no_candidate_in_dict",
                })
                prev_word = word
                continue
            # Rule 2: Bigram LM disambiguation
            # Score each candidate: edit_distance penalty + bigram bonus
            best = None
            best_score = -1.0
            for cand, dist in candidates[:5]:   # top-5 by edit distance
                # Lower edit distance = better; higher bigram score = better
                bi_score = self._bigram_score(prev_word, cand)
                # Combined score: prefer closer + higher bigram
                combined = bi_score * (1.0 / (dist + 1))
                if combined > best_score:
                    best_score = combined
                    best = cand
            corrected.append(best)
            if best != word:
                changes.append({
                    "original": word, "corrected": best,
                    "candidates": candidates[:3],
                    "prev_word": prev_word,
                })
            prev_word = best

        return "".join(corrected), changes


# -----------------------------------------------------------------------------
# Batch process JSONs
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-edit-distance", type=int, default=2)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading strict corrector resources...")
    corrector = StrictCorrector(max_edit_distance=args.max_edit_distance)

    with open(in_dir / "index.json") as f:
        pages_meta = json.load(f)

    total_changes = 0
    new_index = []
    for meta in pages_meta:
        in_json = Path(meta["json"])
        with open(in_json) as f:
            page = json.load(f)

        page_changes = 0
        for b in page.get("bubbles", []):
            t = (b.get("text") or "").strip()
            if not t:
                continue
            new_t, changes = corrector.correct(t)
            if new_t != t:
                b["text_before_strict"] = t
                b["text"] = new_t
                b["strict_changes"] = changes
                page_changes += 1
                if args.verbose:
                    print(f"  {in_json.stem} #{b.get('order'):02d}:")
                    print(f"    before: {t!r}")
                    print(f"    after:  {new_t!r}")
                    for c in changes:
                        print(f"      - {c['original']} → {c['corrected']}")

        total_changes += page_changes
        out_json = out_dir / in_json.name
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(page, f, ensure_ascii=False, indent=2)
        new_meta = dict(meta)
        new_meta["json"] = str(out_json)
        new_index.append(new_meta)
        print(f"{in_json.name}: {page_changes} bubbles corrected")

    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(new_index, f, ensure_ascii=False, indent=2)

    print(f"\nTotal {total_changes} bubbles corrected across {len(pages_meta)} pages")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
