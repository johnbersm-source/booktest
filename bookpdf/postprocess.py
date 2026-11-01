"""후처리: 머리글/바닥글·쪽번호 제거, 인접 페이지 중복 문단 제거, 줄바꿈·하이픈 정리."""
from __future__ import annotations

import re
from collections import Counter

_SENT_END = tuple(".?!。…”\"')」』:;)")
_BLOCK_START = re.compile(r"^(#{1,6}\s|[-*+]\s|\d+[.)]\s|\||\{\{FIG:|>\s)")
_PAGE_NO = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$|^\s*(page|쪽)\s*\d+\s*$", re.I)


def _norm(s: str) -> str:
    return re.sub(r"\W+", "", s).lower()


def _hf_key(b: str) -> str:
    """머리글 비교용 키: 앞/뒤의 '독립된' 숫자(쪽번호)만 지운다. '3장' 같은 글자 붙은 숫자는 유지."""
    return _norm(re.sub(r"^\d+\s+|\s+\d+$", "", b.strip()))


def split_blocks(md: str) -> list[str]:
    return [b.strip("\n") for b in re.split(r"\n\s*\n", md) if b.strip()]


def clean_paragraph(block: str) -> str:
    """문단 내부 줄바꿈 정리. 제목/표/목록/그림 블록은 그대로 둔다."""
    if _BLOCK_START.match(block.lstrip()):
        return block
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif re.search(r"[A-Za-z]-$", out) and re.match(r"[a-z]", ln):
            out = out[:-1] + ln  # 영문 하이픈 줄바꿈 복원
        else:
            out += " " + ln
    return out


def drop_running_headers(pages: list[list[str]]) -> list[list[str]]:
    """여러 페이지 첫/끝 줄에 반복되는 짧은 블록(머리글·바닥글)과 쪽번호 제거."""
    n = len(pages)
    if n >= 4:
        cnt: Counter[str] = Counter()
        for blocks in pages:
            edge = set()
            for b in blocks[:2] + blocks[-2:]:
                if len(b) <= 60 and not b.startswith("|"):
                    edge.add(_hf_key(b))
            cnt.update(k for k in edge if k)
        repeated = {k for k, c in cnt.items() if c >= max(3, int(n * 0.5))}
    else:
        repeated = set()
    out = []
    for blocks in pages:
        keep = []
        for idx, b in enumerate(blocks):
            at_edge = idx < 2 or idx >= len(blocks) - 2
            if _PAGE_NO.match(b):
                continue
            if at_edge and _hf_key(b) in repeated:
                continue
            keep.append(b)
        out.append(keep)
    return out


def dedupe_adjacent(pages: list[list[str]], window: int = 3) -> tuple[list[list[str]], int]:
    """영상 입력에서 같은 페이지가 겹쳐 잡힌 경우 등: 직전 페이지 끝/현재 페이지 앞 문단 중복 제거."""
    removed = 0
    out: list[list[str]] = []
    for blocks in pages:
        if out and out[-1]:
            tail = {_norm(b) for b in out[-1][-window:] if len(_norm(b)) >= 8}
            kept = []
            for j, b in enumerate(blocks):
                if j < window and _norm(b) in tail:
                    removed += 1
                    continue
                kept.append(b)
            blocks = kept
        out.append(blocks)
    return out, removed


def merge_across_pages(pages: list[list[str]]) -> list[str]:
    """페이지를 이어 붙이되, 문장이 쪽 경계에서 끊기면 문단을 합친다."""
    flat: list[str] = []
    for blocks in pages:
        for k, b in enumerate(blocks):
            if (k == 0 and flat and not _BLOCK_START.match(b.lstrip())
                    and not _BLOCK_START.match(flat[-1].lstrip())
                    and not flat[-1].rstrip().endswith(_SENT_END)):
                flat[-1] = flat[-1].rstrip() + " " + b.lstrip()
            else:
                flat.append(b)
    return flat


def postprocess(page_markdowns: list[str]) -> tuple[str, dict]:
    pages = [[clean_paragraph(b) for b in split_blocks(md)] for md in page_markdowns]
    before = sum(len(p) for p in pages)
    pages = drop_running_headers(pages)
    after_hdr = sum(len(p) for p in pages)
    pages, dup = dedupe_adjacent(pages)
    flat = merge_across_pages(pages)
    return "\n\n".join(flat) + "\n", {
        "blocks_in": before, "header_footer_removed": before - after_hdr,
        "duplicates_removed": dup, "blocks_out": len(flat),
    }
