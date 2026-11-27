"""Download, parse, validate and store historical Lotto 6 aus 45 draws from win2day.at.

The download page links one CSV per year (2017+) plus two multi-year archives (1986-2010, 2010-2017).
Layouts differ: archives carry the year in section headers ("2010 Lotto - Beträge in EUR") and dates
without a year ("08.9."); yearly files carry the year only in the file name. Rows with a date are
followed by continuation rows (prize tiers / drawn order) which are ignored.
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import database as db
from database import Draw

logger = logging.getLogger(__name__)

DOWNLOAD_PAGE_URL = "https://www.win2day.at/lotterie/lotto/lotto-statistik-zahlen-ergebnisse-download"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"  # win2day rejects non-browser UAs (HTTP 400)
NUM_MIN, NUM_MAX, NUM_PICK = 1, 45, 6

DATE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.?$")
YEAR_HEADER_RE = re.compile(r"^\s*((?:19|20)\d{2})\s+Lotto", re.IGNORECASE)
FILE_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\.csv$", re.IGNORECASE)
WEEKDAYS = {"mo": 0, "di": 1, "mi": 2, "do": 3, "fr": 4, "sa": 5, "so": 6}


@dataclass
class ParseResult:
    """Draws parsed from one source plus any issues found."""

    draws: list[Draw] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


@dataclass
class CollectionReport:
    """Summary of a collection run."""

    files: int = 0
    parsed: int = 0
    inserted: int = 0
    existing: int = 0
    conflicts: int = 0
    issues: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def make_session(retries: int = 3) -> requests.Session:
    """HTTP session with retry/backoff on transient errors."""
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = USER_AGENT
    return s


def discover_csv_urls(html: str, base_url: str = DOWNLOAD_PAGE_URL) -> list[str]:
    """Extract Lotto CSV download links from the win2day statistics page (deduplicated by path)."""
    soup = BeautifulSoup(html, "lxml")
    seen: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        url = urljoin(base_url, a["href"].strip())
        path = urlparse(url).path
        name = path.rsplit("/", 1)[-1].lower()
        if name.endswith(".csv") and "lotto" in name and "joker" not in name:
            seen.setdefault(path, url)
    return list(seen.values())


def year_from_filename(url_or_name: str) -> Optional[int]:
    """Year encoded in single-year file names like NN_W2D_STAT_Lotto_2024.csv."""
    m = FILE_YEAR_RE.search(urlparse(url_or_name).path or url_or_name)
    return int(m.group(1)) if m else None


def decode_bytes(raw: bytes) -> str:
    """Decode CSV bytes (win2day uses ISO-8859-1/cp1252; newer files may be UTF-8)."""
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def validate_draw(numbers: list[int], additional: Optional[int]) -> list[str]:
    """Return validation errors: exactly 6 distinct ints in 1..45, additional number in 1..45 not among them."""
    errors = []
    if len(numbers) != NUM_PICK:
        errors.append(f"expected {NUM_PICK} numbers, got {len(numbers)}")
    if len(set(numbers)) != len(numbers):
        errors.append(f"duplicate numbers {numbers}")
    if any(not NUM_MIN <= n <= NUM_MAX for n in numbers):
        errors.append(f"number out of range {numbers}")
    if additional is None:
        errors.append("missing additional number")
    elif not NUM_MIN <= additional <= NUM_MAX:
        errors.append(f"additional number out of range {additional}")
    elif additional in numbers:
        errors.append(f"additional number {additional} repeats a main number")
    return errors


def _to_int(cell: str) -> Optional[int]:
    cell = cell.strip()
    return int(cell) if cell.isdigit() else None


def parse_csv_text(text: str, source: str, default_year: Optional[int] = None) -> ParseResult:
    """Parse any known win2day Lotto CSV layout into validated draws."""
    result = ParseResult()
    year = default_year
    delimiter = ";" if text.count(";") >= text.count(",") else ","
    for lineno, row in enumerate(csv.reader(io.StringIO(text), delimiter=delimiter), start=1):
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        header = YEAR_HEADER_RE.match(cells[0])
        if header:
            year = int(header.group(1))
            continue
        date_idx = next((i for i, c in enumerate(cells[:3]) if DATE_RE.match(c)), None)
        if date_idx is None:
            continue
        marker_idx = next((i for i in range(date_idx + 1, len(cells)) if cells[i].lower().startswith("zz")), None)
        before_marker = cells[date_idx + 1:marker_idx] if marker_idx else cells[date_idx + 1:]
        numbers = [n for n in (_to_int(c) for c in before_marker) if n is not None]
        if not numbers:
            continue  # continuation row (prize tiers / payouts)
        if marker_idx is None:
            if len(numbers) >= NUM_PICK:
                result.issues.append(f"{source}:{lineno}: date row without additional-number marker; skipped")
            continue
        additional = next((n for n in (_to_int(c) for c in cells[marker_idx + 1:marker_idx + 3]) if n is not None), None)
        loc = f"{source}:{lineno}"
        if year is None:
            result.issues.append(f"{loc}: cannot determine year; skipped")
            continue
        m = DATE_RE.match(cells[date_idx])
        try:
            d = date(year, int(m.group(2)), int(m.group(1)))
        except ValueError as exc:
            result.issues.append(f"{loc}: invalid date {cells[date_idx]}.{year} ({exc}); skipped")
            continue
        errors = validate_draw(numbers, additional)
        if errors:
            result.issues.append(f"{loc} {d}: invalid draw ({'; '.join(errors)}); skipped")
            continue
        if date_idx > 0:
            wd = WEEKDAYS.get(cells[date_idx - 1].rstrip(".").strip().lower()[:2])
            if wd is not None and wd != d.weekday():
                result.issues.append(f"{loc} {d}: weekday label '{cells[date_idx - 1]}' does not match date")
        result.draws.append(Draw(d, tuple(sorted(numbers)), int(additional), source))
    return result


def deduplicate(draws: Iterable[Draw]) -> tuple[list[Draw], list[str]]:
    """Drop exact duplicates (same date & numbers); drop and report conflicting records for the same date."""
    by_date: dict[date, Draw] = {}
    conflicts: set[date] = set()
    issues: list[str] = []
    dupes = 0
    for d in draws:
        prev = by_date.get(d.draw_date)
        if prev is None:
            by_date[d.draw_date] = d
        elif prev.sorted_numbers == d.sorted_numbers and prev.additional_number == d.additional_number:
            dupes += 1
        else:
            conflicts.add(d.draw_date)
            issues.append(f"conflicting records for {d.draw_date}: {prev.sorted_numbers}+{prev.additional_number} "
                          f"({prev.source}) vs {d.sorted_numbers}+{d.additional_number} ({d.source}); both dropped")
    if dupes:
        issues.append(f"removed {dupes} exact duplicate record(s)")
    unique = [d for k, d in sorted(by_date.items()) if k not in conflicts]
    return unique, issues


def parse_file(path: Path | str) -> ParseResult:
    """Parse a local CSV file."""
    p = Path(path)
    return parse_csv_text(decode_bytes(p.read_bytes()), p.name, year_from_filename(p.name))


def collect(conn, session: Optional[requests.Session] = None, recent_only: bool = False,
            today: Optional[date] = None) -> CollectionReport:
    """Download CSVs from win2day, validate, deduplicate and store new draws.

    recent_only: only fetch single-year files for the current and previous year (used by the scheduler).
    Network errors are raised to the caller after the session's own retries.
    """
    session = session or make_session()
    today = today or date.today()
    report = CollectionReport()
    resp = session.get(DOWNLOAD_PAGE_URL, timeout=30)
    resp.raise_for_status()
    urls = discover_csv_urls(resp.text)
    if not urls:
        raise RuntimeError("No CSV links found on win2day download page - page layout may have changed")
    all_draws: list[Draw] = []
    for url in urls:
        fy = year_from_filename(url)
        if recent_only and (fy is None or fy < today.year - 1):
            continue
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
        except requests.RequestException as exc:
            report.errors.append(f"download failed {url}: {exc}")
            logger.error("Download failed %s: %s", url, exc)
            continue
        name = urlparse(url).path.rsplit("/", 1)[-1]
        parsed = parse_csv_text(decode_bytes(r.content), name, fy)
        report.files += 1
        report.issues.extend(parsed.issues)
        all_draws.extend(parsed.draws)
        logger.info("Parsed %d draws from %s (%d issues)", len(parsed.draws), name, len(parsed.issues))
    unique, dedup_issues = deduplicate(all_draws)
    report.issues.extend(dedup_issues)
    future = [d for d in unique if d.draw_date > today]
    if future:
        report.issues.append(f"{len(future)} draw(s) dated in the future dropped")
        unique = [d for d in unique if d.draw_date <= today]
    report.parsed = len(unique)
    stats = db.insert_draws(conn, unique)
    report.inserted, report.existing, report.conflicts = stats["inserted"], stats["existing"], stats["conflicts"]
    for issue in report.issues:
        logger.warning(issue)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    """CLI: python src/data_collector.py [--recent] [--from-dir DIR] [--db PATH]."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--recent", action="store_true", help="only current/previous year files")
    ap.add_argument("--from-dir", type=Path, help="parse local CSV files instead of downloading")
    ap.add_argument("--db", type=Path, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    conn = db.connect(args.db)
    if args.from_dir:
        draws, issues = [], []
        for f in sorted(args.from_dir.glob("*.csv")):
            res = parse_file(f)
            draws += res.draws
            issues += res.issues
        unique, dd = deduplicate(draws)
        for i in issues + dd:
            logger.warning(i)
        print(db.insert_draws(conn, unique))
        return 0
    rep = collect(conn, recent_only=args.recent)
    print(f"files={rep.files} parsed={rep.parsed} inserted={rep.inserted} existing={rep.existing} "
          f"conflicts={rep.conflicts} issues={len(rep.issues)} errors={len(rep.errors)}")
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
