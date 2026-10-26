"""텍스트 레이어가 있는 PDF 페이지에서 OCR 없이 구조를 뽑는다(네트워크 불필요).

반환 형식은 ocr.py 의 Gemini 결과와 동일:
  {"markdown": str, "figures": [{"box_2d": [ymin,xmin,ymax,xmax](0~1000), "caption": str}],
   "confidence": float, "engine": "textlayer"}
본문에는 그림 자리에 `{{FIG:n}}` 마커를 넣는다.
"""
from __future__ import annotations

import math

import pymupdf as fitz

_SENT_END = tuple(".?!。…”\"')」』:;)")


def _norm_box(rect: fitz.Rect, page: fitz.Rect) -> list[int]:
    w, h = page.width, page.height
    return [
        max(0, int((rect.y0 - page.y0) / h * 1000)),
        max(0, int((rect.x0 - page.x0) / w * 1000)),
        min(1000, math.ceil((rect.y1 - page.y0) / h * 1000)),
        min(1000, math.ceil((rect.x1 - page.x0) / w * 1000)),
    ]


def _md_table(rows: list[list[str | None]]) -> str:
    rows = [[(c or "").replace("\n", " ").replace("|", "\\|").strip() for c in r] for r in rows]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _table_rows(page: fitz.Page, table) -> list[list[str]]:
    """셀 영역을 clip 으로 읽는다(t.extract() 는 쉼표 등 작은 글자를 흩뜨리는 경우가 있었음)."""
    rows = []
    for row in table.rows:
        rows.append([
            page.get_text("text", clip=fitz.Rect(c)).strip() if c else ""
            for c in row.cells
        ])
    return rows


def extract_page(page: fitz.Page) -> dict:
    prect = page.rect
    items: list[tuple[float, str]] = []  # (y, markdown 조각)
    figures: list[dict] = []

    table_rects: list[fitz.Rect] = []
    try:
        for t in page.find_tables().tables:
            r = fitz.Rect(t.bbox)
            table_rects.append(r)
            items.append((r.y0, _md_table(_table_rows(page, t))))
    except Exception:  # noqa: BLE001  (표 탐지 실패는 본문 추출을 막지 않는다)
        pass

    blocks = page.get_text("dict")["blocks"]
    sizes = [
        s["size"]
        for b in blocks if b["type"] == 0
        for ln in b["lines"] for s in ln["spans"] if s["text"].strip()
    ]
    body = sorted(sizes)[len(sizes) // 2] if sizes else 10

    prev_text: list[tuple[float, str]] = []
    prev_head, prev_y1 = False, 0.0
    for b in blocks:
        r = fitz.Rect(b["bbox"])
        if any(r.intersects(t) and (r & t).get_area() > 0.5 * r.get_area() for t in table_rects):
            continue
        if b["type"] == 1:  # 이미지
            if r.width < 30 or r.height < 30:
                continue
            prev_text = []
            figures.append({"box_2d": _norm_box(r, prect), "caption": ""})
            items.append((r.y0, "{{FIG:%d}}" % len(figures)))
            continue
        lines = []
        mx = 0.0
        for ln in b["lines"]:
            txt = "".join(s["text"] for s in ln["spans"]).strip()
            if txt:
                lines.append(txt)
                mx = max(mx, max(s["size"] for s in ln["spans"]))
        if not lines:
            continue
        text = "\n".join(lines)
        is_head = False
        if mx >= body * 1.6 and len(text) < 80:
            text, is_head = "# " + " ".join(lines), True
        elif mx >= body * 1.25 and len(text) < 100:
            text, is_head = "## " + " ".join(lines), True
        # 같은 문단이 PyMuPDF 블록 둘로 쪼개진 경우: 간격이 좁고 앞 블록이 문장 끝이 아니면 이어 붙인다
        if (prev_text and not is_head and not prev_head and r.y0 - prev_y1 < 0.6 * mx
                and not prev_text[-1][1].rstrip().endswith(_SENT_END)):
            prev_text[-1] = (prev_text[-1][0], prev_text[-1][1] + "\n" + text)
            items[-1] = prev_text[-1]
            prev_y1 = r.y1
            continue
        items.append((r.y0, text))
        prev_text, prev_head, prev_y1 = [items[-1]], is_head, r.y1

    items.sort(key=lambda t: t[0])
    return {
        "markdown": "\n\n".join(t for _, t in items if t.strip()),
        "figures": figures,
        "confidence": 0.95 if items else 0.0,
        "engine": "textlayer",
    }
