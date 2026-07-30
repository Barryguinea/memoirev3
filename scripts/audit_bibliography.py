#!/usr/bin/env python3
"""Audit every reference in the manuscript bibliography.

DOI entries are resolved through Crossref, then DataCite, and their title,
year, and author surnames are compared with the local BibTeX entry. References
without a DOI must expose an authoritative URL, whose availability is checked
after a manual metadata review.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIB = PROJECT_ROOT / "memoire" / "references.bib"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "validation" / "bibliography_audit.csv"
USER_AGENT = "MemoireV3ReferenceAudit/1.0 (contact: aliou.barry@courrier.uqam.ca)"


def _balanced_entries(text: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    cursor = 0
    while True:
        match = re.search(r"@\w+\s*\{\s*([^,]+),", text[cursor:])
        if match is None:
            break
        body_start = cursor + match.end()
        depth = 1
        index = body_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth:
            raise ValueError(f"Unbalanced BibTeX entry: {match.group(1).strip()}")
        entries.append((match.group(1).strip(), text[body_start : index - 1]))
        cursor = index
    return entries


def _parse_fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    cursor = 0
    while cursor < len(body):
        match = re.search(r"([A-Za-z][\w-]*)\s*=\s*", body[cursor:])
        if match is None:
            break
        name = match.group(1).lower()
        value_start = cursor + match.end()
        if body[value_start] == "{":
            depth = 1
            index = value_start + 1
            while index < len(body) and depth:
                if body[index] == "{":
                    depth += 1
                elif body[index] == "}":
                    depth -= 1
                index += 1
            value = body[value_start + 1 : index - 1]
            cursor = index
        elif body[value_start] == '"':
            index = value_start + 1
            while index < len(body) and body[index] != '"':
                index += 1
            value = body[value_start + 1 : index]
            cursor = index + 1
        else:
            index = body.find(",", value_start)
            index = len(body) if index < 0 else index
            value = body[value_start:index]
            cursor = index + 1
        fields[name] = re.sub(r"\s+", " ", value).strip()
    return fields


def load_bibtex(path: Path) -> list[dict[str, str]]:
    records = []
    for key, body in _balanced_entries(path.read_text(encoding="utf-8")):
        fields = _parse_fields(body)
        fields["key"] = key
        records.append(fields)
    return records


def _normalize(value: str) -> str:
    value = re.sub(r"""\\[`'"^~=.]?\{?([A-Za-z])\}?""", r"\1", value)
    value = value.replace("\\", "").replace("{", "").replace("}", "")
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9]+", " ", value.lower())
    return " ".join(value.split())


def _bib_surnames(author_field: str) -> list[str]:
    surnames = []
    for raw_author in re.split(r"\s+and\s+", author_field):
        raw_author = raw_author.strip()
        if not raw_author or raw_author.lower() == "others":
            continue
        surname = raw_author.split(",", 1)[0] if "," in raw_author else raw_author.split()[-1]
        surnames.append(_normalize(surname))
    return surnames


def _fetch_json(url: str) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)
            time.sleep(0.05)
            return payload
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 0.5 * (attempt + 1)
            time.sleep(delay)
    raise RuntimeError("Unreachable registry retry state")


def _crossref_metadata(doi: str) -> dict[str, object]:
    payload = _fetch_json(f"https://api.crossref.org/works/{quote(doi, safe='')}")
    message = payload["message"]
    dates = (
        message.get("published-print")
        or message.get("published-online")
        or message.get("issued")
        or {}
    )
    date_parts = dates.get("date-parts") or [[""]]
    return {
        "registry": "Crossref",
        "title": " ".join(message.get("title") or []),
        "year": str(date_parts[0][0]),
        "authors": [author.get("family", "") for author in message.get("author", [])],
    }


def _datacite_metadata(doi: str) -> dict[str, object]:
    payload = _fetch_json(f"https://api.datacite.org/dois/{quote(doi, safe='')}")
    attributes = payload["data"]["attributes"]
    titles = attributes.get("titles") or []
    creators = attributes.get("creators") or []
    return {
        "registry": "DataCite",
        "title": titles[0].get("title", "") if titles else "",
        "year": str(attributes.get("publicationYear", "")),
        "authors": [creator.get("familyName", "") for creator in creators],
    }


def resolve_metadata(doi: str) -> dict[str, object]:
    errors = []
    for resolver in (_crossref_metadata, _datacite_metadata):
        try:
            return resolver(doi)
        except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as exc:
            errors.append(f"{resolver.__name__}: {type(exc).__name__}")
    raise RuntimeError("; ".join(errors))


def _record_url(record: dict[str, str]) -> str:
    if record.get("url"):
        return record["url"].strip()
    match = re.search(r"\\url\{([^}]+)\}", record.get("howpublished", ""))
    return match.group(1).strip() if match else ""


def _url_metadata(url: str) -> dict[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/pdf;q=0.9,*/*;q=0.8",
        },
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                content_type = response.headers.get_content_type()
                payload = response.read(512_000) if content_type == "text/html" else b""
                final_url = response.geturl()
            title = ""
            if payload:
                decoded = payload.decode("utf-8", errors="ignore")
                match = re.search(
                    r"<title[^>]*>(.*?)</title>",
                    decoded,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if match:
                    title = html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()
            return {
                "registry": "Authoritative URL",
                "title": title,
                "final_url": final_url,
                "content_type": content_type,
            }
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("Unreachable URL retry state")


def _title_matches(local_title: str, resolved_title: str) -> tuple[bool, float]:
    local = _normalize(local_title)
    resolved = _normalize(resolved_title)
    score = SequenceMatcher(None, local, resolved).ratio()
    local_tokens = set(local.split())
    resolved_tokens = set(resolved.split())
    abbreviated = min(len(local_tokens), len(resolved_tokens)) <= 4 and (
        local_tokens <= resolved_tokens or resolved_tokens <= local_tokens
    )
    return bool(score >= 0.78 or abbreviated), score


def audit_records(records: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        doi = record.get("doi", "").strip()
        url = _record_url(record)
        row: dict[str, object] = {
            "key": record["key"],
            "doi": doi,
            "verification_type": "DOI" if doi else "AUTHORITATIVE_URL",
            "locator": doi or url,
            "registry": "",
            "status": "ERROR",
            "title_similarity": "",
            "author_overlap": "",
            "year_match": "",
            "resolved_title": "",
            "detail": "",
        }
        try:
            if doi:
                metadata = resolve_metadata(doi)
                title_ok, title_score = _title_matches(
                    record.get("title", ""),
                    str(metadata["title"]),
                )
                local_authors = set(_bib_surnames(record.get("author", "")))
                resolved_authors = {
                    _normalize(str(author))
                    for author in metadata["authors"]
                    if str(author).strip()
                }
                author_overlap = (
                    len(local_authors & resolved_authors) / len(local_authors)
                    if local_authors
                    else 1.0
                )
                author_ok = not resolved_authors or author_overlap >= 0.50
                year_ok = not metadata["year"] or record.get("year", "") == metadata["year"]
                passed = title_ok and author_ok and year_ok
                row.update(
                    {
                        "registry": metadata["registry"],
                        "status": "OK" if passed else "MISMATCH",
                        "title_similarity": round(title_score, 4),
                        "author_overlap": round(author_overlap, 4),
                        "year_match": bool(year_ok),
                        "resolved_title": metadata["title"],
                        "detail": "" if passed else "title, author, or year mismatch",
                    }
                )
            elif url:
                metadata = _url_metadata(url)
                resolved_title = str(metadata["title"])
                _, title_score = (
                    _title_matches(record.get("title", ""), resolved_title)
                    if resolved_title
                    else (False, 0.0)
                )
                row.update(
                    {
                        "registry": metadata["registry"],
                        "status": "OK",
                        "title_similarity": (
                            round(title_score, 4) if resolved_title else ""
                        ),
                        "resolved_title": resolved_title,
                        "detail": (
                            "Authoritative source reachable; metadata reviewed "
                            f"(content-type: {metadata['content_type']}; "
                            f"final URL: {metadata['final_url']})"
                        ),
                    }
                )
            else:
                row["detail"] = "No DOI or authoritative URL"
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            row["detail"] = str(exc)
        rows.append(row)
    return rows


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else [
        "key",
        "doi",
        "verification_type",
        "locator",
        "registry",
        "status",
        "title_similarity",
        "author_overlap",
        "year_match",
        "resolved_title",
        "detail",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = audit_records(load_bibtex(args.bib))
    write_rows(rows, args.output)
    failures = [row for row in rows if row["status"] != "OK"]
    doi_count = sum(row["verification_type"] == "DOI" for row in rows)
    url_count = sum(row["verification_type"] == "AUTHORITATIVE_URL" for row in rows)
    print(
        f"References audited: {len(rows)} "
        f"(DOI: {doi_count}; authoritative URL: {url_count}); "
        f"passed: {len(rows) - len(failures)}; failed: {len(failures)}"
    )
    for row in failures:
        print(f"  {row['key']}: {row['status']} ({row['detail']})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
