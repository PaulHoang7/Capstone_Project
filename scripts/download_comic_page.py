"""
Download comic page images from a website URL.

Supports both static HTML pages (requests + BeautifulSoup) and
JavaScript-rendered pages (Playwright headless Chromium).

Usage:
    # Auto-detect: tries requests first, falls back to Playwright
    python Capstone_project/scripts/download_comic_page.py \
        --url "https://example.com/chapter-1" \
        --out-dir /tmp/comic_pages

    # Force Playwright (for JS-heavy sites like NetTruyen, TruyenQQ)
    python Capstone_project/scripts/download_comic_page.py \
        --url "https://example.com/chapter-1" \
        --out-dir /tmp/comic_pages --browser

    # Download from a list of direct image URLs (one per line)
    python Capstone_project/scripts/download_comic_page.py \
        --url-list urls.txt --out-dir /tmp/comic_pages

    # Filter by minimum dimensions
    python ... --url "..." --min-width 400 --min-height 400
"""

import argparse
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

IMAGE_EXTENSIONS = re.compile(r"\.(jpg|jpeg|png|webp)(\?.*)?$", re.I)


# ─────────────────────────────────────────────────────────────────────────────
# URL extraction — static HTML (requests + BeautifulSoup)
# ─────────────────────────────────────────────────────────────────────────────

def get_image_urls_static(page_url: str) -> list[str]:
    """Extract image URLs from a static HTML page."""
    from bs4 import BeautifulSoup

    resp = requests.get(page_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls = []

    # Common selectors for comic reader sites
    selectors = [
        "div.reading-detail img",       # NetTruyen / TruyenQQ
        "div.chapter-content img",      # Many Vietnamese comic sites
        "div.page-chapter img",         # Alternative layout
        "div#content img",              # Generic reader
        "div.comic-page img",           # Generic
        "img.lazy",                     # Lazy-loaded images
        "img[data-src]",                # Lazy-loaded via data-src
        "img",                          # Fallback: all images
    ]

    for selector in selectors:
        imgs = soup.select(selector)
        if imgs and len(imgs) >= 2:
            # Found a selector with multiple images — likely the comic pages
            for img in imgs:
                src = (img.get("data-src") or img.get("data-original")
                       or img.get("data-lazy-src") or img.get("src"))
                if src:
                    src = urllib.parse.urljoin(page_url, src)
                    src = _upgrade_resolution(src)
                    if src not in urls:
                        urls.append(src)
            if urls:
                break  # Stop at first successful selector

    # If selectors didn't find enough, try all images
    if len(urls) < 2:
        urls = []
        for img in soup.find_all("img"):
            src = (img.get("data-src") or img.get("data-original")
                   or img.get("data-lazy-src") or img.get("src"))
            if not src:
                continue
            src = urllib.parse.urljoin(page_url, src)
            src = _upgrade_resolution(src)
            if src not in urls:
                urls.append(src)

    # Also check <a href="...jpg"> links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if IMAGE_EXTENSIONS.search(href):
            href = urllib.parse.urljoin(page_url, href)
            if href not in urls:
                urls.append(href)

    return urls


# ─────────────────────────────────────────────────────────────────────────────
# URL extraction — Playwright (JS-rendered pages)
# ─────────────────────────────────────────────────────────────────────────────

def get_image_urls_browser(page_url: str) -> list[str]:
    """Extract image URLs from a JS-rendered page using Playwright."""
    from playwright.sync_api import sync_playwright

    print("  [Playwright] Launching headless browser…")
    urls = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Set viewport to desktop size
        page.set_viewport_size({"width": 1920, "height": 1080})
        page.goto(page_url, wait_until="networkidle", timeout=30000)

        # Scroll down to trigger lazy loading
        print("  [Playwright] Scrolling to load lazy images…")
        prev_height = 0
        for _ in range(20):  # max 20 scroll steps
            page.evaluate("window.scrollBy(0, window.innerHeight)")
            time.sleep(0.5)
            height = page.evaluate("document.body.scrollHeight")
            if height == prev_height:
                break
            prev_height = height

        # Wait for images to load
        time.sleep(1)

        # Extract all image sources via JavaScript
        raw_urls = page.evaluate("""
            () => {
                const imgs = document.querySelectorAll('img');
                return Array.from(imgs).map(img => {
                    return img.dataset.src || img.dataset.original
                        || img.dataset.lazySrc || img.currentSrc || img.src;
                }).filter(Boolean);
            }
        """)

        for src in raw_urls:
            src = urllib.parse.urljoin(page_url, src)
            src = _upgrade_resolution(src)
            if src not in urls:
                urls.append(src)

        browser.close()

    return urls


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _upgrade_resolution(url: str) -> str:
    """Try to get the highest resolution version of the image URL."""
    # Blogspot: /sXXX/ → /s0/ (original resolution)
    url = re.sub(r"/s\d+(-[a-z])?/", "/s0/", url)
    # Some CDNs: remove width/height params
    url = re.sub(r"[?&](w|h|width|height)=\d+", "", url)
    return url


def download_images(
    urls: list[str],
    out_dir: Path,
    min_size_bytes: int = 10_000,
    min_width: int = 0,
    min_height: int = 0,
    delay: float = 0.3,
) -> list[Path]:
    """Download images, filtering by size and dimensions."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    for i, url in enumerate(urls):
        # Derive filename
        path_part = urllib.parse.urlparse(url).path
        name = Path(path_part).name or f"image_{i:03d}.jpg"
        name = re.sub(r"[^\w.\-]", "_", name)
        out_path = out_dir / f"{i:03d}_{name}"

        try:
            resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "image" not in content_type and not IMAGE_EXTENSIONS.search(url):
                print(f"  [{i+1:3d}] skip (not image): {url[:70]}")
                continue

            data = resp.content
            if len(data) < min_size_bytes:
                print(f"  [{i+1:3d}] skip ({len(data)//1024}KB < {min_size_bytes//1024}KB): {url[:70]}")
                continue

            # Check image dimensions if filters set
            if min_width > 0 or min_height > 0:
                import cv2
                import numpy as np
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"  [{i+1:3d}] skip (decode failed): {url[:70]}")
                    continue
                h, w = img.shape[:2]
                if w < min_width or h < min_height:
                    print(f"  [{i+1:3d}] skip ({w}×{h} too small): {url[:70]}")
                    continue

            out_path.write_bytes(data)
            print(f"  [{i+1:3d}] saved {len(data)//1024:4d}KB → {out_path.name}")
            saved.append(out_path)

        except Exception as e:
            print(f"  [{i+1:3d}] ERROR: {e}  ({url[:70]})")

        if delay > 0:
            time.sleep(delay)

    return saved


def load_url_list(path: str) -> list[str]:
    """Load image URLs from a text file (one URL per line)."""
    urls = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Download comic images from a URL (static or JS-rendered)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--url", help="Page URL to scrape images from")
    p.add_argument("--url-list", help="Text file with direct image URLs (one per line)")
    p.add_argument("--out-dir", default="/tmp/comic_pages",
                   help="Output directory")
    p.add_argument("--min-size-kb", type=int, default=20,
                   help="Minimum image size in KB")
    p.add_argument("--min-width", type=int, default=0,
                   help="Minimum image width in pixels (0=no filter)")
    p.add_argument("--min-height", type=int, default=0,
                   help="Minimum image height in pixels (0=no filter)")
    p.add_argument("--browser", action="store_true",
                   help="Force Playwright (headless Chromium) for JS-rendered sites")
    p.add_argument("--delay", type=float, default=0.3,
                   help="Delay between downloads (seconds)")
    args = p.parse_args()

    if not args.url and not args.url_list:
        p.error("Provide --url or --url-list")

    out_dir = Path(args.out_dir)

    if args.url_list:
        print(f"Loading URLs from: {args.url_list}")
        urls = load_url_list(args.url_list)
    elif args.browser:
        print(f"Fetching (Playwright): {args.url}")
        urls = get_image_urls_browser(args.url)
    else:
        # Try static first, fall back to browser if too few images
        print(f"Fetching (static): {args.url}")
        urls = get_image_urls_static(args.url)
        if len(urls) < 3:
            print(f"  Only {len(urls)} image(s) found. Retrying with Playwright…")
            try:
                urls = get_image_urls_browser(args.url)
            except Exception as e:
                print(f"  Playwright failed: {e}")

    print(f"Found {len(urls)} image URL(s)\n")

    if not urls:
        print("No images found. Tips:")
        print("  1. Try --browser flag for JS-rendered sites")
        print("  2. Try --url-list with direct CDN URLs")
        sys.exit(1)

    saved = download_images(
        urls, out_dir,
        min_size_bytes=args.min_size_kb * 1024,
        min_width=args.min_width,
        min_height=args.min_height,
        delay=args.delay,
    )

    print(f"\nDownloaded {len(saved)} image(s) → {out_dir}")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
