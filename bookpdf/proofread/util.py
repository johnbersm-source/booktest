"""교정 파이프라인 공용 도구: 쪽 입출력, 토큰화, 자모 분해, 편집거리.

이 모듈은 외부 서비스·API 를 쓰지 않는다. 쪽 데이터는 bookpdf 가 만든 work/pages/pNNNN.json 을 그대로 읽는다.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------- 쪽 입출력

PAGE_JSON = re.compile(r"^p(\d{4,})\.json$")


def load_pages(work: Path, only: list[int] | None = None, subdir: str = "pages") -> list[dict]:
    """work/pages/pNNNN.json → [{page, markdown, image, engine, blocks, json_path, raw}] (쪽 번호순).

    markdown 은 json 의 'markdown' 필드. 필드가 없으면 같은 이름의 .md 를 읽는다.
    only 가 있으면 그 쪽 번호만(없는 번호는 조용히 무시하지 않고 호출자가 확인할 수 있게 missing 으로 알린다)."""
    res_dir = Path(work) / subdir if subdir else Path(work)
    if not res_dir.is_dir():
        raise FileNotFoundError(f"쪽 결과 폴더가 없습니다: {res_dir} (먼저 python -m bookpdf 로 쪽별 결과를 만드세요)")
    found: dict[int, Path] = {}
    for p in res_dir.iterdir():
        m = PAGE_JSON.match(p.name)
        if m:
            found[int(m.group(1))] = p
    nums = sorted(found)
    if only is not None:
        want = set(only)
        nums = [n for n in nums if n in want]
    pages = []
    for n in nums:
        raw = json.loads(found[n].read_text(encoding="utf-8"))
        md = raw.get("markdown")
        if md is None:
            mdp = found[n].with_suffix(".md")
            md = mdp.read_text(encoding="utf-8") if mdp.exists() else ""
        pages.append({"page": n, "markdown": md, "image": raw.get("image"), "engine": raw.get("engine", ""),
                      "blocks": raw.get("blocks"), "page_number": raw.get("page_number"),
                      "confidence": raw.get("confidence"), "json_path": str(found[n]), "raw": raw})
    return pages


def missing_pages(pages: list[dict], only: list[int] | None) -> list[int]:
    have = {p["page"] for p in pages}
    return [n for n in (only or []) if n not in have]


def lines_of(md: str) -> list[str]:
    """쪽 Markdown → 줄 목록(빈 줄 포함, 줄 번호는 1부터 = 인덱스+1). 줄 수를 바꾸는 수정은 하지 않는다."""
    return md.split("\n")


FIG_MARK = re.compile(r"\{\{FIG:[^}]*\}\}")
UNREADABLE = "[판독불가]"

# ---------------------------------------------------------------- 문자 종류

def is_hangul_syllable(ch: str) -> bool:
    return "가" <= ch <= "힣"


def has_hangul(s: str) -> bool:
    return any(is_hangul_syllable(c) or "ㄱ" <= c <= "ㆎ" for c in s)


def has_latin(s: str) -> bool:
    return any(("a" <= c <= "z") or ("A" <= c <= "Z") for c in s)


def has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)


# ---------------------------------------------------------------- 토큰화

@dataclass(frozen=True)
class Tok:
    text: str        # 공백으로 구분된 원 토큰(구두점 포함)
    start: int       # 줄 안 시작 위치
    end: int
    core: str        # 앞뒤 구두점을 뗀 핵심 글자
    cstart: int      # core 의 줄 안 시작 위치
    cend: int


_WS_TOK = re.compile(r"\S+")
_EDGE_PUNCT = re.compile(r"^[^\w가-힣]+|[^\w가-힣%]+$")


def tokenize_line(line: str) -> list[Tok]:
    out = []
    for m in _WS_TOK.finditer(line):
        t = m.group(0)
        i = 0
        while i < len(t) and not (t[i].isalnum() or is_hangul_syllable(t[i])):
            i += 1
        j = len(t)
        while j > i and not (t[j - 1].isalnum() or is_hangul_syllable(t[j - 1]) or t[j - 1] == "%"):
            j -= 1
        out.append(Tok(t, m.start(), m.end(), t[i:j], m.start() + i, m.start() + j))
    return out


_WORD_RE = re.compile(r"[가-힣]+|[A-Za-z][A-Za-z0-9&]*|\d[\d,.]*\d|\d")


def iter_words(line: str):
    """줄에서 (종류, 글자, 시작, 끝) 을 순서대로. 종류: ko(한글 연속), en(영문+숫자), num(숫자열)."""
    for m in _WORD_RE.finditer(line):
        t = m.group(0)
        kind = "ko" if is_hangul_syllable(t[0]) else ("en" if t[0].isalpha() else "num")
        yield kind, t, m.start(), m.end()


# ---------------------------------------------------------------- 자모

_L = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_V = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_T = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"


def to_jamo(s: str) -> str:
    """한글 음절을 호환 자모 열로 분해(받침 없음은 생략). 한글이 아닌 글자는 그대로."""
    out = []
    for ch in unicodedata.normalize("NFC", s):
        if is_hangul_syllable(ch):
            c = ord(ch) - 0xAC00
            out.append(_L[c // 588])
            out.append(_V[(c % 588) // 28])
            t = c % 28
            if t:
                out.append(_T[t])
        else:
            out.append(ch)
    return "".join(out)


# 모양이 비슷해 OCR 이 잘 혼동하는 자모 쌍(대칭). 치환 비용을 낮춰 후보 순위에 반영한다. (경험적 목록 — 실제 책으로 미검증)
_CONFUSABLE_PAIRS = [
    "ㅏㅓ", "ㅑㅕ", "ㅗㅜ", "ㅛㅠ", "ㅐㅔ", "ㅒㅖ", "ㅏㅑ", "ㅓㅕ", "ㅗㅛ", "ㅜㅠ", "ㅣㅡ", "ㅡㅜ", "ㅡㅗ",
    "ㄱㅋ", "ㅅㅈ", "ㅈㅊ", "ㅂㅍ", "ㅇㅎ", "ㅁㅇ", "ㄴㄷ", "ㄷㄹ", "ㅁㅂ", "ㄱㄴ", "ㅅㅆ", "ㅐㅏ", "ㅔㅓ",
    "ㅏㅣ", "ㅓㅣ", "ㅗㅏ", "ㅜㅓ", "ㅢㅟ", "ㅙㅐ", "ㅚㅐ",
]
CONFUSABLE = {frozenset(p) for p in _CONFUSABLE_PAIRS}
CONFUSABLE_COST = 0.4


def levenshtein(a, b) -> int:
    """일반 편집거리(삽입·삭제·치환 비용 1)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def jamo_distance(a: str, b: str) -> int:
    return levenshtein(to_jamo(a), to_jamo(b))


def weighted_jamo_distance(a: str, b: str) -> float:
    """자모 열 편집거리. 모양이 비슷한 자모끼리의 치환은 0.4 로 계산한다."""
    x, y = to_jamo(a), to_jamo(b)
    if x == y:
        return 0.0
    prev = [float(j) for j in range(len(y) + 1)]
    for i, ca in enumerate(x, 1):
        cur = [float(i)]
        for j, cb in enumerate(y, 1):
            if ca == cb:
                sub = 0.0
            elif frozenset((ca, cb)) in CONFUSABLE:
                sub = CONFUSABLE_COST
            else:
                sub = 1.0
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub))
        prev = cur
    return prev[-1]


def myers_distance(a, b) -> int:
    """Myers/Hyyrö 비트병렬 편집거리. 긴 글(쪽 단위 CER/WER)도 빠르게. a, b 는 같은 종류의 시퀀스(문자열 또는 리스트)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a            # b(패턴)를 짧은 쪽으로
    m = len(b)
    peq: dict = {}
    for i, c in enumerate(b):
        peq[c] = peq.get(c, 0) | (1 << i)
    mask = (1 << m) - 1
    pv, mv, score = mask, 0, m
    top = 1 << (m - 1)
    for c in a:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | ~(xh | pv)
        mh = pv & xh
        if ph & top:
            score += 1
        elif mh & top:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = (mh | ~(xv | ph)) & mask
        mv = ph & xv
    return score


# ---------------------------------------------------------------- 조사(한국어 어미) 분리

# 흔한 조사. 긴 것부터 검사한다. (완전하지 않음: 어미·복합 조사는 후보 생성에서 어간을 못 찾으면 그냥 통째로 비교)
JOSA = sorted([
    "에서는", "에서의", "으로는", "으로의", "으로서", "으로써", "에게서", "에서도", "까지는", "부터는", "이라는", "이라고", "이라면",
    "에서", "에게", "한테", "으로", "까지", "부터", "보다", "처럼", "만큼", "이나", "이란", "이며", "이고", "이다", "에는", "에도",
    "와의", "과의", "으론", "라는", "라고", "라면", "하고",
    "은", "는", "이", "가", "을", "를", "의", "에", "도", "와", "과", "로", "만", "나", "란", "며", "고", "다",
], key=len, reverse=True)


def split_josa(core: str) -> tuple[str, str]:
    """(어간, 조사). 어간이 2글자 미만이 되면 분리하지 않는다."""
    if not core or not is_hangul_syllable(core[-1]):
        return core, ""
    for j in JOSA:
        if core.endswith(j) and len(core) - len(j) >= 2:
            return core[: -len(j)], j
    return core, ""


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default
