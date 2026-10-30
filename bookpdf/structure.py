"""tesseract 전체 책 모드: 모든 쪽의 블록을 모아 머리글 제거 → 제목 판정 → 그림 위주 쪽 원본 삽입 → 목차.

입력은 ocr_tesseract 가 저장한 쪽별 결과(blocks/chars/ink/page_number). 이 단계는 OCR 을 다시 하지 않으므로
옵션(--embed-*)만 바꿔 빠르게 다시 조립할 수 있다.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .postprocess import _hf_key

_CHAPTER = re.compile(r"^\s*(제\s*\d+\s*[장부편절화]|chapter\s+\d+|part\s+\d+|\d+\s*장\b)", re.I)
_SECTION = re.compile(r"^\s*\d+\.\d+(\.\d+)?\s+\S")
_SENT_END = (".", "。", "!", "?", ",", ";", "다.", "요.")
EMBED_MAX_SIDE = 1600
MIN_HEADING_CONF = 70     # 제목 후보 블록의 단어 신뢰도(0~100) 하한


@dataclass
class EmbedOptions:
    enabled: bool = False
    min_chars: int = 120      # 이 글자 수 미만이면 '글자가 적은 쪽'
    min_ink: float = 0.04     # 배경과 다른 화소 비율이 이 이상이어야 '그림이 있는 쪽'(빈 쪽은 제외)


def repeated_edge_keys(results: list[dict], min_pages: int = 3) -> set[str]:
    """위/아래 구역에 서로 다른 min_pages 쪽 이상 반복되는 짧은 블록 = 머리글·바닥글."""
    seen: dict[str, set[int]] = {}
    for r in results:
        for b in r.get("blocks", []):
            if b["zone"] == "body" or len(b["text"]) > 80:
                continue
            k = _hf_key(b["text"].replace("\n", " "))
            if len(k) >= 4:
                seen.setdefault(k, set()).add(r["page"])
    return {k for k, pages in seen.items() if len(pages) >= min_pages}


def body_line_height(results: list[dict]) -> float:
    """본문 글자 높이: 글자 수로 가중한 중앙값(본문 구역·20자 이상 블록만)."""
    vals: list[float] = []
    for r in results:
        for b in r.get("blocks", []):
            if b["zone"] == "body" and len(b["text"]) >= 20:
                vals.extend([b["lh"]] * max(1, len(b["text"]) // 10))
    return statistics.median(vals) if vals else 0.0


def heading_level(b: dict, body_h: float, min_ratio: float = 1.25) -> int:
    """0=본문, 1=대제목(#), 2=중제목(##). 짧고 문장부호로 끝나지 않으며 글자가 본문보다 클 때만."""
    text = b["text"].replace("\n", " ").strip()
    if not text or len(text) > 60 or b["lines"] > 2 or text.endswith(_SENT_END) or b["zone"] != "body":
        return 0
    if b.get("conf", 100) < MIN_HEADING_CONF or len(re.sub(r"\W", "", text)) < 3:
        return 0        # 깨진 조각(낮은 신뢰도·너무 짧음)은 글자가 커 보여도 제목으로 보지 않는다
    ratio = b["lh"] / body_h if body_h else 1.0
    if ratio >= min_ratio * 1.2 or (_CHAPTER.match(text) and ratio >= 1.1):
        return 1
    if ratio >= min_ratio or (_SECTION.match(text) and ratio >= 1.1):
        return 2
    return 0


def is_isolated(b: dict, others: list[dict], body_h: float) -> bool:
    """제목은 위아래가 비어 있다: 같은 줄이나 바로 인접한 줄(본문 글자 높이의 절반 이내)에 다른 본문 블록이 없어야 한다.
    문단 한가운데 깨져 나온 조각(글자가 커 보이는 오인식)을 걸러낸다."""
    pad = 0.5 * body_h
    for o in others:
        if o is b or o["zone"] != "body":
            continue
        if o["y0"] < b["y1"] + pad and o["y1"] > b["y0"] - pad:
            return False
    return True


def is_picture_page(r: dict, opt: EmbedOptions) -> bool:
    return bool(opt.enabled and r.get("chars", 0) < opt.min_chars and r.get("ink", 0.0) >= opt.min_ink)


def make_embed_image(src: Path, dst: Path) -> Path:
    """쪽 원본을 책에 넣기 좋은 크기의 JPEG 로 저장(이미 있으면 재사용)."""
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            s = EMBED_MAX_SIDE / max(im.size)
            if s < 1:
                im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
            im.save(dst, quality=82)
    return dst


def estimate_printed_pages(results: list[dict]) -> dict[int, tuple[int, bool]]:
    """쪽 순번 → (인쇄 쪽번호, 추정여부). 읽힌 번호가 없는 쪽은 가장 가까운 읽힌 쪽에서 순번 차이만큼 보정."""
    known = [(r["page"], r["page_number"]) for r in results if r.get("page_number")]
    out: dict[int, tuple[int, bool]] = {}
    if not known:
        return out
    for r in results:
        p = r["page"]
        j, n = min(known, key=lambda t: abs(t[0] - p))
        out[p] = (n + (p - j), j != p)
    return out


def assemble(results: list[dict], opt: EmbedOptions, embed_dir: Path, heading_ratio: float = 1.25
             ) -> tuple[list[str], dict[str, dict], list[dict], dict]:
    """반환: (쪽별 markdown, 삽입 그림 맵, 목차 항목, 통계).

    목차 항목: {level, text, page(순번), printed(인쇄 쪽번호 또는 None), est(추정 여부)}
    """
    rep = repeated_edge_keys(results)
    body_h = body_line_height(results)
    printed = estimate_printed_pages(results)
    mds: list[str] = []
    figs: dict[str, dict] = {}
    toc: list[dict] = []
    stats = {"edge_removed": 0, "headings": 0, "picture_pages": 0, "body_line_height": round(body_h, 1)}
    for r in results:
        pic = is_picture_page(r, opt)
        parts: list[str] = []
        for b in r.get("blocks", []):
            if b["zone"] != "body" and _hf_key(b["text"].replace("\n", " ")) in rep:
                stats["edge_removed"] += 1
                continue
            lvl = heading_level(b, body_h, heading_ratio) if is_isolated(b, r.get("blocks", []), body_h) else 0
            if lvl:
                text = b["text"].replace("\n", " ").strip()
                parts.append("#" * lvl + " " + text)
                pn, est = printed.get(r["page"], (None, False))
                toc.append({"level": lvl, "text": text, "page": r["page"], "printed": pn, "est": est})
                stats["headings"] += 1
            elif not pic:
                parts.append(b["text"])       # 그림 위주 쪽은 제목만 남기고 나머지 글자(축 라벨 등 오인식)는 원본 이미지에 맡긴다
        if pic:
            img = Path(r["image"])
            key = f"full{r['page']}"
            dst = make_embed_image(img, embed_dir / f"p{r['page']:04d}.jpg")
            figs[key] = {"path": str(dst), "caption": "", "frac_w": 1.0}
            parts.append("{{FIG:%s}}" % key)
            stats["picture_pages"] += 1
        mds.append("\n\n".join(parts))
    return mds, figs, toc, stats


def match_toc_to_body(toc: list[dict], body: str) -> list[dict]:
    """후처리 후에도 본문에 남은 제목만 목차에 둔다(순서 유지, 같은 제목이 여러 번 나와도 순서대로 대응)."""
    heads = [(len(m.group(1)), m.group(2).strip()) for m in re.finditer(r"^(#{1,2})\s+(.+)$", body, re.M)]
    out, i = [], 0
    for lvl, text in heads:
        while i < len(toc) and not (toc[i]["level"] == lvl and toc[i]["text"] == text):
            i += 1
        if i == len(toc):
            break
        out.append(toc[i])
        i += 1
    return out
