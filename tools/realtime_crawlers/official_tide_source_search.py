"""Search official portals for tide-gauge / storm-surge data pages.

This is a lightweight crawler for source discovery. It does not bypass login,
CAPTCHA, or API authorization. It only downloads public HTML pages and records
pages whose title/text/URL contains tide-related keywords.

Default domains:
- Taiwan CWA Open Data portal
- Taiwan government open data portal
- China NMEFC
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen

from runtime_paths import TIDE_SOURCE_MATCHES_PATH


DEFAULT_START_URLS = [
    "https://opendata.cwa.gov.tw/",
    "https://data.gov.tw/",
    "https://www.nmefc.cn/",
]

DEFAULT_KEYWORDS = [
    "潮位",
    "潮高",
    "潮汐",
    "水位",
    "验潮",
    "驗潮",
    "潮位站",
    "浮标",
    "浮標",
    "海象",
    "风暴潮",
    "風暴潮",
    "storm surge",
    "tide",
    "water level",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


class LinkTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag.lower() == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        clean = re.sub(r"\s+", " ", data).strip()
        if not clean:
            return
        self.text_parts.append(clean)
        if self._in_title:
            self.title_parts.append(clean)

    @property
    def title(self) -> str:
        return " ".join(self.title_parts).strip()

    @property
    def text(self) -> str:
        return " ".join(self.text_parts).strip()


def normalize_url(base_url: str, href: str) -> str:
    absolute = urljoin(base_url, href)
    absolute, _fragment = urldefrag(absolute)
    return absolute


def same_allowed_domain(url: str, allowed_domains: set[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return host in allowed_domains


def fetch_html(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"})
    with urlopen(request, timeout=30) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""
        raw = response.read()
    return raw.decode("utf-8", errors="replace")


def matched_keywords(*values: str, keywords: Iterable[str]) -> list[str]:
    haystack = " ".join(values).lower()
    return [keyword for keyword in keywords if keyword.lower() in haystack]


def crawl(start_urls: list[str], keywords: list[str], max_pages: int) -> list[dict[str, str]]:
    allowed_domains = {urlparse(url).netloc.lower() for url in start_urls}
    queue: deque[str] = deque(start_urls)
    seen: set[str] = set()
    matches: list[dict[str, str]] = []

    while queue and len(seen) < max_pages:
        url = queue.popleft()
        if url in seen:
            continue
        seen.add(url)

        try:
            html = fetch_html(url)
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            print(f"skip: {url} ({exc})")
            continue
        if not html:
            continue

        parser = LinkTextParser()
        parser.feed(html)
        hits = matched_keywords(url, parser.title, parser.text[:5000], keywords=keywords)
        if hits:
            matches.append(
                {
                    "url": url,
                    "title": parser.title,
                    "keywords": ";".join(hits),
                    "snippet": parser.text[:300],
                }
            )
            print(f"match: {url} [{';'.join(hits)}]")

        for href in parser.links:
            next_url = normalize_url(url, href)
            parsed = urlparse(next_url)
            if parsed.scheme not in ("http", "https"):
                continue
            if not same_allowed_domain(next_url, allowed_domains):
                continue
            if next_url not in seen:
                queue.append(next_url)

    return matches


def write_matches(matches: list[dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["url", "title", "keywords", "snippet"]
    with out_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(matches)
    print(f"saved: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search official portals for tide-gauge data sources.")
    parser.add_argument("--start-url", action="append", default=[], help="Official portal start URL")
    parser.add_argument("--keyword", action="append", default=[], help="Keyword to match")
    parser.add_argument("--max-pages", type=int, default=80, help="Maximum pages to fetch")
    parser.add_argument(
        "--out",
        default=str(TIDE_SOURCE_MATCHES_PATH),
        help="Output CSV path",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    start_urls = args.start_url or DEFAULT_START_URLS
    keywords = args.keyword or DEFAULT_KEYWORDS
    matches = crawl(start_urls, keywords, args.max_pages)
    write_matches(matches, Path(args.out))


if __name__ == "__main__":
    main()
