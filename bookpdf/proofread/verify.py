"""M5 이미지 근거 검증: 의심 줄을 쪽 이미지에서 잘라 Gemini 에게 '독립적으로' 다시 읽게 한다.

독립성: 프롬프트에 OCR 글자·후보·제안을 넣지 않는다(page_anchor 모드의 앞뒤 이웃 낱말만 예외 — 대상 낱말 자체는 절대 넣지 않음).
채택 규칙: 재판독 결과가 M3/M4 후보 중 정확히 하나를 포함하고 원문(OCR 토큰)은 포함하지 않을 때만 'confirmed'.
  · 재판독에 원문이 그대로 있고 후보는 없음  → supports_original (오탐으로 보고 수정하지 않음, dismissed 로 집계)
  · 어느 후보와도 안 맞음/모호함/[판독불가]    → mismatch / unreadable (보류)
  · 이미지 없음, 자르기 불가, 호출 실패        → no_image / error (보류)
자르기: 단어 상자(tsv) > 블록 안 줄 위치 추정(tesseract blocks) > 쪽 전체 이미지 + 앞뒤 낱말 안내(page_anchor).
"""
from __future__ import annotations

import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from . import util
from .cache import ResponseCache, key_of

SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

PROMPT_CROP = """이 이미지는 책 한 쪽에서 잘라낸 일부(보통 한 줄 또는 몇 낱말)입니다. 이미지에 보이는 글자를 순서대로 그대로 옮겨 적으세요.
규칙: 한글·영문·숫자·기호를 보이는 그대로 적고, 맞춤법을 고치거나 뜻을 풀거나 보이지 않는 글자를 추가하지 마세요.
읽을 수 없는 글자는 [판독불가] 로 표시하세요. 출력은 JSON 만: {"text": "옮겨 적은 글자"}"""

PROMPT_PAGE = """이 이미지는 책의 한 쪽입니다. 이 쪽에서 왼쪽 글자 "{left}" 바로 다음부터 오른쪽 글자 "{right}" 바로 앞까지의 글자를 보이는 그대로 옮겨 적으세요.
(왼쪽/오른쪽 글자는 위치 안내용이며 결과에 포함하지 마세요.)
규칙: 맞춤법을 고치거나 보이지 않는 글자를 추가하지 마세요. 읽을 수 없으면 [판독불가]. 출력은 JSON 만: {{"text": "옮겨 적은 글자"}}"""


def squash(s: str) -> str:
    """비교용: NFC, 공백·구두점 제거."""
    s = unicodedata.normalize("NFC", s)
    return "".join(c for c in s if c.isalnum() or util.is_hangul_syllable(c))


def block_line_map(blocks: list[dict]) -> list[tuple[int, int, int] | None]:
    """tesseract blocks → 마크다운 줄 번호(0부터)별 (블록 번호, 블록 안 줄 번호, 블록 줄 수). 빈 줄은 None."""
    out: list[tuple[int, int, int] | None] = []
    for bi, b in enumerate(blocks):
        n = b["text"].count("\n") + 1
        out += [(bi, k, n) for k in range(n)]
        if bi < len(blocks) - 1:
            out.append(None)
    return out


def locate_crop(page: dict, span: dict, size: tuple[int, int]) -> tuple[tuple[int, int, int, int] | None, str]:
    """(자를 상자 px, 모드). 상자가 None 이면 page_anchor 모드."""
    W, H = size
    box = span.get("box")
    if box:
        x0, y0, x1, y1 = box
        h = max(y1 - y0, 8)
        c = (max(0, x0 - int(2.5 * h)), max(0, y0 - int(0.4 * h)), min(W, x1 + int(2.5 * h)), min(H, y1 + int(0.4 * h)))
        return c, "word_box"
    blocks = page.get("blocks")
    md = page.get("markdown_original")
    if blocks and md is not None:
        lm = block_line_map(blocks)
        if len(lm) == len(util.lines_of(md)) and span["line"] - 1 < len(lm) and lm[span["line"] - 1]:
            bi, k, n = lm[span["line"] - 1]
            b = blocks[bi]
            lh = (b["y1"] - b["y0"]) / n
            y0 = b["y0"] + k * lh
            c = (max(0, int(b["x0"] - 0.5 * lh)), max(0, int(y0 - 0.35 * lh)), min(W, int(b["x1"] + 0.5 * lh)), min(H, int(y0 + 1.35 * lh)))
            if c[2] - c[0] > 8 and c[3] - c[1] > 8:
                return c, "line_est"
    return None, "page_anchor"


def _png(im: Image.Image, max_side: int = 1600, min_h: int = 48) -> bytes:
    im = im.convert("RGB")
    if im.height < min_h:                                  # 너무 작은 줄은 확대해야 잘 읽는다
        f = min(4.0, min_h / max(im.height, 1))
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    if max(im.size) > max_side:
        f = max_side / max(im.size)
        im = im.resize((int(im.width * f), int(im.height * f)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _anchors(line: str, span: dict) -> tuple[str, str]:
    left = line[: span["col"]].split()[-2:]
    right = line[span["end"]:].split()[:2]
    return " ".join(left), " ".join(right)


@dataclass
class VerifyStats:
    calls_planned: int = 0
    cache_hits: int = 0
    confirmed: int = 0
    mismatch: int = 0
    unreadable: int = 0
    supports_original: int = 0
    no_image: int = 0
    error: int = 0
    stopped: str | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def decide(reread: str, token: str, cand_texts: list[str]) -> tuple[str, str | None]:
    """재판독 글자 ↔ 원문/후보 비교 → (verdict, 일치한 후보)."""
    rr = squash(reread)
    if not rr or "판독불가" in reread:
        return "unreadable", None
    tok = squash(token)
    hits = []
    for c in cand_texts:
        sc = squash(c)
        if len(sc) >= 1 and sc in rr:
            hits.append(c)
    # 다른 후보에 포함되는 짧은 후보는 제외(예: '거래' vs '거래량')
    hits = [c for c in hits if not any(c != d and squash(c) in squash(d) for d in hits)]
    tok_in = bool(tok) and tok in rr
    if hits and not tok_in:
        if len(hits) == 1:
            return "confirmed", hits[0]
        return "mismatch", None                 # 후보 둘 이상이 동시에 맞음: 모호
    if tok_in and not hits:
        return "supports_original", None
    return "mismatch", None


def verify_spans(spans: list[dict], cands: dict[str, list[str]], pages: dict[int, dict], caller: Callable[..., str],
                 crops_dir: Path, cache: ResponseCache | None = None, log: Callable[[str], None] = print
                 ) -> tuple[dict[str, dict], VerifyStats]:
    """spans 각각을 이미지로 재판독. cands[span id] = 비교할 후보 글자 목록(M3+M4). 반환: ({id: 결과}, 통계)."""
    from ..ocr import QuotaExhausted
    st = VerifyStats()
    out: dict[str, dict] = {}
    crops_dir = Path(crops_dir)
    crops_dir.mkdir(parents=True, exist_ok=True)
    for s in spans:
        texts = cands.get(s["id"], [])
        rec = {"id": s["id"], "mode": None, "reread": None, "verdict": None, "matched": None, "crop": None}
        out[s["id"]] = rec
        pg = pages.get(s["page"])
        if not texts:
            rec["verdict"] = "mismatch"
            rec["note"] = "비교할 후보 없음"
            st.mismatch += 1
            continue
        if not pg or not pg.get("image") or not Path(pg["image"]).exists():
            rec["verdict"] = "no_image"
            st.no_image += 1
            continue
        with Image.open(pg["image"]) as im:
            im.load()
            box, mode = locate_crop(pg, s, im.size)
            rec["mode"] = mode
            if box:
                data = _png(im.crop(box))
                prompt = PROMPT_CROP
                crop_path = crops_dir / f"{s['id']}.png"
                crop_path.write_bytes(data)
                rec["crop"] = crop_path.name
            else:
                left, right = _anchors(pg["lines"][s["line"] - 1], s)
                data = _png(im)
                prompt = PROMPT_PAGE.format(left=left or "(줄 처음)", right=right or "(줄 끝)")
        ck = key_of("verify", prompt, data)
        text = cache.get(ck) if cache else None
        if text is not None:
            st.cache_hits += 1
        else:
            try:
                st.calls_planned += 1
                text = caller(prompt, image=data, mime="image/png", schema=SCHEMA)
                if cache:
                    cache.put(ck, text)
            except QuotaExhausted as e:
                st.stopped = str(e)
                log(f"[이미지 검증 중단] {e}")
                rec["verdict"] = "error"
                rec["note"] = "호출 한도로 중단"
                st.error += 1
                for r in spans[spans.index(s) + 1:]:
                    out[r["id"]] = {"id": r["id"], "mode": None, "reread": None, "verdict": "error", "matched": None,
                                    "crop": None, "note": "호출 한도로 중단"}
                    st.error += 1
                return out, st
            except Exception as e:  # noqa: BLE001
                rec["verdict"] = "error"
                rec["note"] = str(e)[:160]
                st.error += 1
                continue
        try:
            t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            reread = json.loads(t).get("text") if t.startswith("{") else t
            if not isinstance(reread, str):
                raise ValueError("text 필드 없음")
        except (ValueError, AttributeError):
            rec["verdict"] = "error"
            rec["note"] = "응답 형식 오류"
            st.error += 1
            continue
        rec["reread"] = reread
        verdict, matched = decide(reread, s["token"], texts)
        rec["verdict"], rec["matched"] = verdict, matched
        setattr(st, verdict, getattr(st, verdict) + 1)
    return out, st
