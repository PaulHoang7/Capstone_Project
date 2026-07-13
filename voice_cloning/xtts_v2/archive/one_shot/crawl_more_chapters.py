"""Crawl chap 2+ for existing 5 series + chap 1 of Shin-chan + Slam Dunk.

Strategy: thuviensach.vn uses sequential /img/comic/<slug>/img_XXXXX.webp.
We pull a large range per series and delete failed/small downloads.
"""
import subprocess
from pathlib import Path

BASE_DIR = Path("/mnt/nfs-data/tin_dataset/comic/vietnamese")

# (folder name, thuviensach path slug, start idx, count)
CRAWL_JOBS = [
    # Existing 5 series — add chap 2 (continue img range)
    ("doraemon_ch2",     "Doraemon-Dai-Tuyen-Tap", 40, 80),    # ch1 ended ~38
    ("conan_ch2",        "Conan",                   16, 80),    # ch1 = img 0-15
    ("dragonball_ch2",   "Dragon-Ball",             37, 80),    # ch1 = img 0-36
    ("onepiece_ch2",     "One-Piece",               54, 80),    # ch1 = img 2-53
    ("naruto_ch2",       "Naruto",                  19, 80),    # ch1 = img 0-18
    # New series — chap 1
    ("shinchan_ch1",     "Shin-Cau-Be-But-Chi",      0, 80),
    ("slamdunk_ch1",     "Cao-Thu-Bong-Ro-Slam-Dunk", 0, 80),
]


def crawl_series(folder_name: str, slug: str, start: int, count: int):
    out_dir = BASE_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {folder_name} ({slug}, img_{start}..{start+count-1}) ===")
    downloaded = 0
    for n in range(start, start + count):
        idx = f"{n:05d}"
        url = f"https://thuviensach.vn/img/comic/{slug}/img_{idx}.webp"
        out_file = out_dir / f"{folder_name}_p{n:04d}.webp"
        if out_file.exists():
            continue
        # curl silently, check http code
        result = subprocess.run(
            ["curl", "-s", "-o", str(out_file),
             "-A", "Mozilla/5.0",
             "-e", "https://thuviensach.vn/",
             "-w", "%{http_code}",
             url],
            capture_output=True, text=True, timeout=30,
        )
        http_code = result.stdout.strip()
        # Keep only valid downloads (200 + not tiny)
        if http_code != "200" or out_file.stat().st_size < 20_000:
            out_file.unlink(missing_ok=True)
        else:
            downloaded += 1
    print(f"  Downloaded: {downloaded}")
    return downloaded


def main():
    total = 0
    for folder, slug, start, count in CRAWL_JOBS:
        total += crawl_series(folder, slug, start, count)
    print(f"\n=== Summary ===\nTotal new pages: {total}")
    # Print all series pages count
    print("\nAll series pages:")
    for d in sorted(BASE_DIR.iterdir()):
        if d.is_dir():
            n = len(list(d.glob("*.webp"))) + len(list(d.glob("*.jpg")))
            print(f"  {d.name}: {n} pages")


if __name__ == "__main__":
    main()
