"""Corpus acquisition and parsing.

Downloads the two NIST PDFs and a sample of CISA ICS advisories, then loads
everything into a flat list of document units tagged with ``doc_type`` so the
chunker can treat standards and advisories differently.

All network/PDF libraries are imported lazily inside functions so the module
(and the test suite) imports without them installed.
"""
from __future__ import annotations

import json
import pathlib
from typing import Callable

from .config import Config

HEADERS = {"User-Agent": "Mozilla/5.0 (SecureOps assistant)"}

# Used only when the live CISA fetch fails, so the pipeline always runs.
FALLBACK_ADVISORIES = [
    {"title": "ICSA-FALLBACK-01: Example PLC Hardcoded Credentials", "url": "fallback://01",
     "text": "Example PLC family, CVSS 9.8. Hardcoded credentials (CWE-798) let a remote "
             "attacker modify control logic. Mitigations: update firmware, isolate control "
             "networks behind firewalls, use VPNs for remote access."},
    {"title": "ICSA-FALLBACK-02: Example HMI Path Traversal", "url": "fallback://02",
     "text": "Example HMI product, path traversal (CWE-22), CVSS 7.5, allows reading arbitrary "
             "files. Mitigations: upgrade, restrict network access, monitor file access, apply "
             "defense-in-depth."},
    {"title": "ICSA-FALLBACK-03: Example Historian SQL Injection", "url": "fallback://03",
     "text": "Example historian server, SQL injection (CWE-89), CVSS 8.6, in the web reporting "
             "interface. Mitigations: apply hotfix, enforce least privilege, audit logs, segment "
             "historians in a DMZ."},
]


def download_pdfs(cfg: Config, log: Callable[[str], None] = print) -> pathlib.Path:
    """Download each configured PDF once (skip if already on disk)."""
    import requests

    d = pathlib.Path(cfg.corpus.dir)
    d.mkdir(exist_ok=True)
    for fname, url in cfg.corpus.pdfs.items():
        dest = d / fname
        if dest.exists():
            log(f"exists: {fname}")
            continue
        r = requests.get(url, headers=HEADERS, timeout=120)
        r.raise_for_status()
        dest.write_bytes(r.content)
        log(f"saved {fname} ({len(r.content)/1e6:.1f} MB)")
    return d


def fetch_cisa(cfg: Config, log: Callable[[str], None] = print) -> list[dict]:
    """Fetch CISA ICS advisories from the RSS feed; fall back to bundled samples."""
    import time

    import requests
    from bs4 import BeautifulSoup

    advisories: list[dict] = []
    try:
        feed = requests.get(cfg.corpus.cisa_feed_url, headers=HEADERS, timeout=60)
        feed.raise_for_status()
        items = BeautifulSoup(feed.content, "xml").find_all("item")[: cfg.corpus.cisa_n_advisories]
        for it in items:
            title, link = it.title.get_text(strip=True), it.link.get_text(strip=True)
            try:
                page = requests.get(link, headers=HEADERS, timeout=60)
                page.raise_for_status()
                soup = BeautifulSoup(page.content, "lxml")
                main = soup.find("main") or soup.body
                text = " ".join(main.get_text(" ", strip=True).split())
                if len(text) > 500:
                    advisories.append({"title": title, "url": link, "text": text})
                time.sleep(1)  # be polite to CISA
            except Exception as e:  # one bad page shouldn't sink the run
                log(f"skip {link}: {e}")
    except Exception as e:
        log(f"feed error: {e}")

    if len(advisories) < 3:
        log("using bundled fallback advisories")
        advisories = FALLBACK_ADVISORIES

    out = pathlib.Path(cfg.corpus.dir) / "cisa_advisories.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(advisories, indent=2))
    return advisories


def load_documents(cfg: Config) -> list[dict]:
    """Return document units: {source, page, doc_type, text} for PDFs + advisories."""
    from pypdf import PdfReader

    d = pathlib.Path(cfg.corpus.dir)
    docs: list[dict] = []

    for fname in cfg.corpus.pdfs:
        reader = PdfReader(str(d / fname))
        for i, pg in enumerate(reader.pages, start=1):
            text = (pg.extract_text() or "").strip()
            if len(text) > 80:  # skip near-empty pages
                docs.append({"source": fname, "page": i, "doc_type": "nist",
                             "text": " ".join(text.split())})

    adv_path = d / "cisa_advisories.json"
    if adv_path.exists():
        for adv in json.loads(adv_path.read_text()):
            docs.append({"source": f"CISA: {adv['title']}", "page": 1,
                         "doc_type": "advisory", "text": adv["text"]})
    return docs
