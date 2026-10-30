"""글꼴에 없는 글자 처리.

NanumGothic 에 없는 글자는 xhtml2pdf 가 빈칸으로 그린다. 글자별로 다음 순서로 처리한다.
  1. 원문자(①❶㉠ 등)  → (1) (가) 같은 안전한 문자로 치환 (Noto 에 글리프가 있어도 치환: 글꼴 의존을 줄임)
  2. NanumGothic 에 있음 → 그대로
  3. Noto Sans CJK KR(번들 부분집합)에 있음(한자·CJK 기호) → <span class="cjk"> 로 감싸 그 글꼴로 표시
  4. 둘 다 없음 → □ 로 대체(빈칸이 되어 글자가 사라진 것을 모르는 일을 막음)하고 리포트에 목록으로 남김
"""
from __future__ import annotations

import html
import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

from reportlab.pdfbase.ttfonts import TTFont

from .config import FONT_DIR, FONT_REGULAR

CJK_REGULAR = FONT_DIR / "NotoSansCJKkr-Regular.ttf"
CJK_BOLD = FONT_DIR / "NotoSansCJKkr-Bold.ttf"
PLACEHOLDER = "□"
_HANGUL_CIRCLED = "가나다라마바사아자차카타파하"


def _build_circled() -> dict[str, str]:
    m: dict[str, str] = {}
    for i in range(1, 21):
        m[chr(0x2460 + i - 1)] = f"({i})"            # ①–⑳
    for i in range(21, 36):
        m[chr(0x3251 + i - 21)] = f"({i})"           # ㉑–㉟
    for i in range(36, 51):
        m[chr(0x32B1 + i - 36)] = f"({i})"           # ㊱–㊿
    for i in range(1, 11):
        m[chr(0x2776 + i - 1)] = f"({i})"            # ❶–❿ (검은 원문자)
        m[chr(0x2780 + i - 1)] = f"({i})"            # ➀–➉
        m[chr(0x278A + i - 1)] = f"({i})"            # ➊–➓
        m[chr(0x2474 + i - 1)] = f"({i})"            # ⑴–⒇ 중 1–10 (괄호 숫자)
    for i in range(26):
        m[chr(0x24B6 + i)] = f"({chr(65 + i)})"      # Ⓐ–Ⓩ
        m[chr(0x24D0 + i)] = f"({chr(97 + i)})"      # ⓐ–ⓩ
    for i, ch in enumerate(_HANGUL_CIRCLED):
        m[chr(0x3260 + i)] = f"({ch})"               # ㉠–㉭
        m[chr(0x326E + i)] = f"({ch})"               # ㉮–㉻
    m["⓿"] = "(0)"
    m["㈜"] = "(주)"
    return m


CIRCLED = _build_circled()
_TOKEN = re.compile("|".join(re.escape(k) for k in sorted(CIRCLED, key=len, reverse=True)))


@lru_cache(maxsize=1)
def _cmaps() -> tuple[frozenset[int], frozenset[int]]:
    nanum = frozenset(TTFont("nanum_cov", str(FONT_REGULAR)).face.charToGlyph)
    cjk = frozenset(TTFont("cjk_cov", str(CJK_REGULAR)).face.charToGlyph) if CJK_REGULAR.exists() else frozenset()
    return nanum, cjk


def _is_ignorable(ch: str) -> bool:
    """글리프가 필요 없는 문자(공백·제어·서식 문자·결합 부호 등)."""
    return ch in "\n\r\t" or unicodedata.category(ch) in ("Cc", "Cf", "Zs", "Zl", "Zp", "Mn") or ch == "️"


def classify(ch: str) -> str:
    """'ok' | 'circled' | 'cjk' | 'missing' | 'ignore'"""
    if _is_ignorable(ch):
        return "ignore"
    if ch in CIRCLED:
        return "circled"
    nanum, cjk = _cmaps()
    if ord(ch) in nanum:
        return "ok"
    if ord(ch) in cjk:
        return "cjk"
    return "missing"


def normalize(text: str) -> str:
    """원문자 → (n) 치환, NanumGothic·Noto 어느 쪽에도 없는 글자 → □. 한자는 그대로(나중에 span)."""
    text = _TOKEN.sub(lambda m: CIRCLED[m.group(0)], text)
    return "".join(PLACEHOLDER if classify(c) == "missing" else c for c in text)


_TAG = re.compile(r"(<[^>]*>)")


def wrap_cjk(html_fragment: str) -> str:
    """HTML 조각의 '글자 부분'에서 Noto 로만 그릴 수 있는 연속 글자를 <span class="cjk"> 로 감싼다(태그 속은 건드리지 않음)."""
    nanum, cjk = _cmaps()

    def conv(seg: str) -> str:
        out, run = [], []
        for ch in seg:
            if ord(ch) not in nanum and ord(ch) in cjk and not _is_ignorable(ch):
                run.append(ch)
                continue
            if run:
                out.append('<span class="cjk">' + "".join(run) + "</span>")
                run = []
            out.append(ch)
        if run:
            out.append('<span class="cjk">' + "".join(run) + "</span>")
        return "".join(out)

    return "".join(p if p.startswith("<") and p.endswith(">") else conv(p) for p in _TAG.split(html_fragment))


def text_to_html(text: str) -> str:
    """일반 텍스트(표지 제목 등) → 정규화 + 이스케이프 + span."""
    return wrap_cjk(html.escape(normalize(text)))


def scan(pages: list[tuple[int, str]]) -> dict:
    """(쪽번호, 원문) 목록을 훑어 글꼴에 없는 글자를 모은다. 글리프 채움이 일어나기 전의 원문 기준.

    반환: {"circled": {글자: [횟수, [쪽...]]}, "cjk": {...}, "missing": {...}}
    """
    out: dict[str, dict[str, list]] = {"circled": {}, "cjk": {}, "missing": {}}
    for page, text in pages:
        text = re.sub(r"\{\{FIG:[\w\-]+\}\}", "", text)
        for ch in text:
            k = classify(ch)
            if k in ("circled", "cjk", "missing"):
                e = out[k].setdefault(ch, [0, []])
                e[0] += 1
                if page not in e[1]:
                    e[1].append(page)
    return out


def format_report(found: dict, max_rows: int = 40) -> list[str]:
    lines = ["", "## 글꼴에 없는 글자 (NanumGothic 기준)", ""]
    total = sum(len(v) for v in found.values())
    if total == 0:
        return lines + ["- 없음(모든 글자가 NanumGothic 으로 표시됨)."]
    if not CJK_REGULAR.exists():
        lines.append(f"- **경고**: 번들 글꼴 {CJK_REGULAR.name} 이 없어 한자·CJK 기호가 □ 로 나갑니다.")

    def rows(kind: str, how: str) -> None:
        items = sorted(found[kind].items(), key=lambda kv: (-kv[1][0], kv[0]))
        if not items:
            return
        lines.append(f"### {how} — {len(items)}종")
        for ch, (n, pages) in items[:max_rows]:
            pg = ", ".join(f"p{p}" for p in pages[:6]) + (" …" if len(pages) > 6 else "")
            shown = f"{ch} → {CIRCLED[ch]}" if kind == "circled" else ch
            lines.append(f"- {shown} (U+{ord(ch):04X}) ×{n}  {pg}")
        if len(items) > max_rows:
            lines.append(f"- … 외 {len(items) - max_rows}종 (`리포트`에는 상위 {max_rows}종만 표시)")
        lines.append("")

    rows("cjk", "한자·CJK 기호 → Noto Sans CJK KR 로 표시")
    rows("circled", "원문자 → 괄호 숫자/글자로 대체")
    rows("missing", f"어느 글꼴에도 없음 → {PLACEHOLDER} 로 대체 (원문 확인 필요)")
    return lines
