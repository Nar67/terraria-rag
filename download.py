#!/usr/bin/env python3
"""Download the Terraria Wiki (terraria.wiki.gg) and save each page's text.

Usage:
    python3 download.py                 # everything, about 4,000 pages (~70 min)
    python3 download.py --limit 20      # a short test run
    python3 download.py --list          # print the page list and exit

Follows the wiki's robots.txt: the page list comes from the published
sitemap (the API and Special: pages are off-limits), only /wiki/ article
pages are fetched, one request per --delay seconds (default 1).

For each page it writes text/<Title>.md (extract.py's output, with the
source URL and licence under the title) and adds a line to manifest.jsonl.
Run it again at any time: pages whose text file is newer than the page's
last edit are skipped, so an interrupted run resumes and a later run only
fetches pages that changed. --keep-html also stores the page's HTML,
gzipped, in html/; a page then counts as up to date only once its HTML is
saved. After changing extract.py, --from-html rebuilds text/ from html/
without downloading anything.

Content licence: CC BY-NC-SA 4.0 (credit the Terraria Wiki, non-commercial,
share alike). The site's signals forbid AI training on it (ai-train=no).
"""
import argparse
import gzip
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import extract  # noqa: E402  (extract.py sits next to this script)

BASE = "https://terraria.wiki.gg"
SITEMAP_INDEX = f"{BASE}/sitemaps/sitemap-index-terraria_en.xml"
# Namespaces taken from the sitemap: 0 = articles, 10000 = Guide:.
# Left out: 11000 = Legacy: (old game versions), 14 = Category: (lists).
NAMESPACES = {"0", "10000"}
LICENSE = "CC BY-NC-SA 4.0 (Terraria Wiki contributors)"
LANG_CODE = re.compile(r"^[a-z]{2,3}(-[a-z]{2,4})?$")


class Blocked(Exception):
    pass


class Fetcher:
    def __init__(self, user_agent, delay):
        self.user_agent = user_agent
        self.delay = delay
        self.last = 0.0
        self.forbidden = 0

    def get(self, url, tries=5):
        """GET with the rate limit, gzip, and backoff on 429/5xx. Returns
        (bytes, final url) or None for a 404."""
        wait = 30
        for attempt in range(1, tries + 1):
            pause = self.delay - (time.monotonic() - self.last)
            if pause > 0:
                time.sleep(pause)
            self.last = time.monotonic()
            req = urllib.request.Request(url, headers={
                "User-Agent": self.user_agent, "Accept-Encoding": "gzip"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                    if r.headers.get("Content-Encoding") == "gzip" or data[:2] == b"\x1f\x8b":
                        data = gzip.decompress(data)
                    self.forbidden = 0
                    return data, r.geturl()
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code == 403:
                    self.forbidden += 1
                    if self.forbidden >= 3:
                        raise Blocked("three 403 answers in a row: the site is refusing us, stopping")
                    return None
                if e.code in (429, 500, 502, 503, 504) and attempt < tries:
                    retry = e.headers.get("Retry-After")
                    sleep_for = int(retry) if retry and retry.isdigit() else wait
                    print(f"    HTTP {e.code}, waiting {sleep_for}s", file=sys.stderr)
                    time.sleep(sleep_for)
                    wait *= 2
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < tries:
                    print(f"    {e}, waiting {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    wait *= 2
                    continue
                raise
        return None


def wanted(title):
    """Skip translations and template fragments (lower-case subpages of
    articles); keep content subpages and every Guide: section."""
    if "/" not in title:
        return True
    last = title.rsplit("/", 1)[1]
    if LANG_CODE.match(last) or last == "Crafting_tree":  # the tree is a drawing; trees come from recipes
        return False
    if title.startswith("Guide:"):
        return True
    return last[:1].isupper() or last[:1].isdigit()


def page_list(fetcher):
    index, _ = fetcher.get(SITEMAP_INDEX)
    sitemaps = re.findall(r"<loc>(.*?)</loc>", index.decode())
    pages = []
    for sm in sitemaps:
        m = re.search(r"-NS_(\d+)-\d+\.xml", sm)
        if not m or m.group(1) not in NAMESPACES:
            continue
        data, _ = fetcher.get(sm)
        xml = data.decode()
        for block in re.findall(r"<url>(.*?)</url>", xml, re.S):
            loc = re.search(r"<loc>(.*?)</loc>", block)
            mod = re.search(r"<lastmod>(.*?)</lastmod>", block)
            if not loc or "/wiki/" not in loc.group(1):
                continue
            title = urllib.parse.unquote(loc.group(1).split("/wiki/", 1)[1])
            if wanted(title):
                lastmod = datetime.fromisoformat(mod.group(1).replace("Z", "+00:00")) if mod else None
                pages.append((title, loc.group(1), lastmod))
    return pages


def file_name(title):
    # "/" would make a folder; keep everything else readable.
    return title.replace("/", "%2F").replace("\0", "")


def atomic_write(path, data):
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def page_text(html, url):
    """extract.py's Markdown with the source URL and licence under the title."""
    page_title, text = extract(html.decode("utf-8", errors="replace"))
    header, _, body = text.partition("\n")
    return page_title, f"{header}\n\nSource: {url}\nLicense: {LICENSE}\n{body}"


def rebuild_from_html(out):
    """Re-extract every saved html/*.html.gz into text/, without the network."""
    html_dir, text_dir = out / "html", out / "text"
    files = sorted(html_dir.glob("*.html.gz"))
    if not files:
        sys.exit("no saved HTML in html/; download once with --keep-html first")
    urls = {}  # file name -> source URL, latest manifest entry wins
    manifest = out / "manifest.jsonl"
    if manifest.exists():
        for line in manifest.open(encoding="utf-8"):
            entry = json.loads(line)
            urls[file_name(entry["title"])] = entry["url"]
    text_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        name = f.name[:-len(".html.gz")]
        url = urls.get(name) or f"{BASE}/wiki/" + urllib.parse.quote(
            name.replace("%2F", "/").replace(" ", "_"), safe="/:'(),!")
        _, text = page_text(gzip.decompress(f.read_bytes()), url)
        atomic_write(text_dir / (name + ".md"), text.encode("utf-8"))
        if i % 500 == 0:
            print(f"{i}/{len(files)} pages", file=sys.stderr)
    print(f"re-extracted {len(files)} pages into text/", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description="Download the Terraria Wiki as text.")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                    help="project folder (default: this script's folder)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (minimum 1)")
    ap.add_argument("--limit", type=int, default=0, help="stop after fetching this many pages")
    ap.add_argument("--keep-html", action="store_true", help="also save the HTML, gzipped, in html/")
    ap.add_argument("--contact", default="", help="an email or URL added to the User-Agent")
    ap.add_argument("--list", action="store_true", help="print the page list and exit")
    ap.add_argument("--from-html", action="store_true",
                    help="re-extract text/ from the saved html/ only; no downloading")
    args = ap.parse_args()

    if args.from_html:
        rebuild_from_html(args.out)
        return

    delay = max(1.0, args.delay)
    agent = "terraria-rag-learning/0.1 (personal learning project"
    agent += f"; {args.contact})" if args.contact else ")"
    fetcher = Fetcher(agent, delay)

    pages = page_list(fetcher)
    if args.list:
        for title, url, lastmod in pages:
            print(title)
        print(f"{len(pages)} pages", file=sys.stderr)
        return

    text_dir = args.out / "text"
    html_dir = args.out / "html"
    text_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_html:
        html_dir.mkdir(parents=True, exist_ok=True)
    manifest = (args.out / "manifest.jsonl").open("a", encoding="utf-8")

    todo = []
    for title, url, lastmod in pages:
        path = text_dir / (file_name(title) + ".md")
        # With --keep-html a page is only up to date once its HTML is saved too,
        # so the first such run fetches every page (and resumes if interrupted).
        check = html_dir / (file_name(title) + ".html.gz") if args.keep_html else path
        if check.exists() and lastmod and check.stat().st_mtime >= lastmod.timestamp():
            continue
        todo.append((title, url, path))
    up_to_date = len(pages) - len(todo)
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(pages)} pages listed, {up_to_date} up to date, "
          f"{len(todo)} to fetch now (about {len(todo) * delay / 60:.0f} min)", file=sys.stderr)

    done = failed = 0
    start = time.monotonic()
    try:
        for i, (title, url, path) in enumerate(todo, 1):
            try:
                got = fetcher.get(url)
            except Blocked:
                raise
            except Exception as e:  # one bad page must not stop the run
                print(f"[{i}/{len(todo)}] {title}: failed ({e})", file=sys.stderr)
                failed += 1
                continue
            if got is None:
                print(f"[{i}/{len(todo)}] {title}: not found, skipped", file=sys.stderr)
                failed += 1
                continue
            html, final_url = got
            page_title, text = page_text(html, url)
            atomic_write(path, text.encode("utf-8"))
            if args.keep_html:  # written last: its time marks the page as done
                atomic_write(html_dir / (file_name(title) + ".html.gz"), gzip.compress(html))
            manifest.write(json.dumps({
                "title": title, "page_title": page_title, "url": url, "final_url": final_url,
                "file": str(path.relative_to(args.out)), "bytes": len(text.encode()),
                "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")
            manifest.flush()
            done += 1
            if i % 25 == 0 or i == len(todo):
                left = (time.monotonic() - start) / i * (len(todo) - i)
                print(f"[{i}/{len(todo)}] {title}  ({left / 60:.0f} min left)", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nstopped; run again to continue where it left off", file=sys.stderr)
    except Blocked as e:
        print(f"\n{e}", file=sys.stderr)
    finally:
        manifest.close()
    print(f"saved {done} pages, {failed} failed or missing", file=sys.stderr)


if __name__ == "__main__":
    main()
