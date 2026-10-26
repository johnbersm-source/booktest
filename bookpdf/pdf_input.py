"""PDF 입력 어댑터: PyMuPDF로 페이지를 이미지로 변환."""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

MAX_SIDE = 2600  # 긴 변 상한(px). Gemini 전송 크기/속도 균형


def validate_pdf(path: Path) -> int:
    """입력 검증. 페이지 수를 돌려주고, 문제가 있으면 ValueError."""
    if not path.is_file():
        raise ValueError(f"파일이 없습니다: {path}")
    try:
        doc = fitz.open(path)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"PDF를 열 수 없습니다: {path} ({e})") from e
    if doc.needs_pass:
        raise ValueError("암호가 걸린 PDF는 지원하지 않습니다.")
    n = doc.page_count
    doc.close()
    if n == 0:
        raise ValueError("페이지가 0개인 PDF입니다.")
    return n


def render_pages(pdf: Path, out_dir: Path, dpi: int = 200) -> list[Path]:
    """페이지별 PNG 렌더. 이미 있는 파일은 건너뛴다(재개용)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf)
    paths: list[Path] = []
    for i, page in enumerate(doc, start=1):
        p = out_dir / f"p{i:04d}.png"
        paths.append(p)
        if p.exists():
            continue
        scale = dpi / 72
        longest = max(page.rect.width, page.rect.height) * scale
        if longest > MAX_SIDE:
            scale *= MAX_SIDE / longest
        page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).save(p)
    doc.close()
    return paths


def has_text_layer(pdf: Path, min_chars: int = 30) -> list[bool]:
    doc = fitz.open(pdf)
    flags = [len(pg.get_text("text").strip()) >= min_chars for pg in doc]
    doc.close()
    return flags
