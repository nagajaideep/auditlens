"""Download public documents for the RAG corpus into docs/policies/public/.

By default it fetches a company's latest 10-K from SEC EDGAR. The SEC asks automated clients to send a
descriptive User-Agent that includes contact details, so pass your own with --user-agent.
You can also download PCAOB / IAASB standards as PDFs by hand and drop them in the same folder.

    python scripts/fetch_public_docs.py --cik 320193 --user-agent "Your Name you@example.com"
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "policies" / "public"


def get(url: str, ua: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua, "Accept-Encoding": "identity"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def latest_10k(cik: str, ua: str) -> Path:
    cik = cik.zfill(10)
    sub = json.loads(get(f"https://data.sec.gov/submissions/CIK{cik}.json", ua))
    recent = sub["filings"]["recent"]
    i = recent["form"].index("10-K")
    acc, doc = recent["accessionNumber"][i].replace("-", ""), recent["primaryDocument"][i]
    html = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}", ua).decode("utf-8", "ignore")
    text = re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style).*?</\1>", " ", html))
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    # turn "Item 1A." style headings into markdown headings so chunks carry section citations
    text = re.sub(r"\n\s*(Item\s+\d+[A-Z]?\.)", r"\n## \1", text)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{sub['name'].replace(' ', '_')}_10-K_{recent['filingDate'][i]}.md"
    out.write_text(f"# {sub['name']} Form 10-K ({recent['filingDate'][i]})\n\n{text}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cik", default="320193", help="SEC CIK (default 320193)")
    ap.add_argument("--user-agent", required=True, help="Your name and email, as SEC requires")
    a = ap.parse_args()
    print("saved", latest_10k(a.cik, a.user_agent))
