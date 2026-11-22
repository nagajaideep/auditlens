"""Load policy documents (.md / .txt / .pdf) and split them into cited chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    section: str
    page: int | None = None

    @property
    def citation(self) -> str:
        loc = f"p.{self.page}" if self.page else self.section
        return f"{self.source} — {loc}"


def _read(path: Path) -> list[tuple[str, int | None]]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return [(p.extract_text() or "", i + 1) for i, p in enumerate(PdfReader(str(path)).pages)]
    return [(path.read_text(encoding="utf-8", errors="ignore"), None)]


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown-ish text on headings; returns (heading, body)."""
    parts, heading, buf = [], "Introduction", []
    for line in text.splitlines():
        m = re.match(r"^\s*#{1,4}\s+(.*)", line)
        if m:
            if "".join(buf).strip():
                parts.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if "".join(buf).strip():
        parts.append((heading, "\n".join(buf).strip()))
    return parts


def _window(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(" ".join(words)) <= size:
        return [" ".join(words)]
    out, cur = [], []
    for w in words:
        cur.append(w)
        if len(" ".join(cur)) >= size:
            out.append(" ".join(cur))
            keep, n = [], 0
            for x in reversed(cur):
                n += len(x) + 1
                if n > overlap:
                    break
                keep.insert(0, x)
            cur = keep
    if cur and (not out or " ".join(cur) not in out[-1]):
        out.append(" ".join(cur))
    return out


def chunk_documents(docs_dir: str | Path, size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(docs_dir).glob("**/*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"} or path.name.lower() == "readme.md":
            continue
        for page_text, page in _read(path):
            for heading, body in _sections(page_text):
                for i, piece in enumerate(_window(body, size, overlap)):
                    cid = f"{path.stem}:{page or 0}:{len(chunks)}:{i}"
                    chunks.append(Chunk(cid, f"{heading}\n{piece}", path.name, heading, page))
    return chunks
