"""M1 정규화: 결정적이고 되돌릴 수 있는 표기 통일. 내용(단어)은 바꾸지 않는다.

적용 규칙(규칙 id → 설명):
  nfc          유니코드 NFC 조합, 폭 0 글자·소프트 하이픈 제거, NBSP/전각 공백 → 공백
  fullwidth    전각 ASCII·전각 기호 → 반각(ＡＢＣ１２３，→ ABC123,). 글리프가 없어지는 반각 한글/가타카나는 제외
  quote        따옴표 통일(“ ” → ",  ‘ ’ → ')   ※ quote_style="keep" 이면 건너뜀
  hyphen       하이픈·마이너스 변종(‐ ‑ ‒ − ﹣ －) → -   (en/em dash 는 건드리지 않음)
  space        연속 공백 압축, 줄 끝 공백 제거, 문장부호 앞 공백 제거("다 ." → "다.")
  digit_conf   숫자 문맥에서만 O/o→0, l/I/|→1 ("2O21년" → "2021년"), 숫자 뒤 rn → m ("5rn" → "5m")

보호: 용어집(glossary.txt + 자동 수집)과 영문 약어 화이트리스트(MACD, RSI, KOSPI …)에 있는 토큰은 digit_conf 를 적용하지 않는다.
모든 변경은 줄 단위 (이전, 이후, 규칙) 로 기록한다 → 되돌릴 수 있음.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import util

# 금융·경제 책에서 흔한 영문 약어. 완전한 목록이 아니므로 acronyms.txt 로 추가하세요.
DEFAULT_ACRONYMS = frozenset("""
MACD RSI KOSPI KOSDAQ KRX KRW USD JPY EUR CNY PER PBR PSR PCR EPS BPS DPS ROE ROA ROIC EBIT EBITDA EV
ETF ETN ELS DLS ELW REITs REIT GDP GNP CPI PPI PCE PMI FOMC FED FRB ECB BOK BOJ PBOC IMF OECD WTO BIS OPEC
IPO MBO LBO DCF WACC CAGR YoY QoQ MoM NAV VIX NASDAQ NYSE DJIA OTC CFD DART HTS MTS API AI IT ICT IoT 5G 6G
ESG CEO CFO CTO COO R&D M&A S&P TV PC USB LED OLED LCD DRAM NAND CPU GPU HBM SSD OCR PDF SW HW BM VC PE NPL EV
""".split())

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
SPACE_VARIANTS = dict.fromkeys(map(ord, " 　    "), " ")
QUOTE_MAP = str.maketrans({"“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                           "‘": "'", "’": "'", "‚": "'", "‛": "'"})
HYPHEN_MAP = str.maketrans({c: "-" for c in "‐‑‒−﹣－"})
DIGIT_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})


def _FULLWIDTH_OK(c: str) -> bool:
    """NFKC 로 바꿔도 NanumGothic/Noto 에 글리프가 있는 전각 글자만(전각 ASCII U+FF01~FF5E, 전각 기호 ￠￡￢￤￥￦).
    반각 한글 자모(→ 조합용 자모, 글리프 없음)·반각 가타카나·￣(→ 공백+결합 문자)는 건드리지 않는다."""
    o = ord(c)
    return 0xFF01 <= o <= 0xFF5E or o in (0xFFE0, 0xFFE1, 0xFFE2, 0xFFE4, 0xFFE5, 0xFFE6)


UNIT_FOLLOW = set("%년월일원억조만천백개명배위점호권부원달러")


# ------------------------------------------------------------------ 보호 목록
class Protect:
    """용어집 + 영문 약어. 이 목록의 토큰은 자동 규칙·수정 대상에서 보호한다."""

    def __init__(self, terms=(), acronyms=DEFAULT_ACRONYMS):
        self.terms = {t for t in terms if t}
        self.acronyms = set(acronyms)

    @classmethod
    def load(cls, glossary: Path | None = None, acronyms: Path | None = None, auto: Path | None = None) -> "Protect":
        terms: set[str] = set()
        for p in (glossary, auto):
            if p and Path(p).exists():
                terms |= set(read_term_file(Path(p)))
        acr = set(DEFAULT_ACRONYMS)
        if acronyms and Path(acronyms).exists():
            acr |= set(read_term_file(Path(acronyms)))
        return cls(terms, acr)

    def __contains__(self, s: str) -> bool:
        return s in self.terms or s in self.acronyms

    def all_terms(self) -> set[str]:
        return self.terms | self.acronyms


def read_term_file(path: Path) -> list[str]:
    """한 줄에 용어 하나(탭 뒤 숫자는 빈도로 보고 무시). '#' 로 시작하면 주석."""
    out = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        t = raw.split("\t")[0].strip()
        if t and not t.startswith("#"):
            out.append(unicodedata.normalize("NFC", t))
    return out


def collect_glossary(texts: list[str], min_count: int = 5, min_pages: int = 3) -> list[tuple[str, int, int]]:
    """책 본문에서 자주 나오는 용어(한글 어간·영문 단어)를 모은다 → [(용어, 횟수, 쪽수)] 횟수 내림차순.

    주의: 같은 오인식이 여러 번 반복되면 그 오타도 '자주 나오는 용어'로 잡혀 보호될 수 있다.
    그래서 기준을 높게(기본 5회·3쪽 이상) 두고, 결과는 사람이 검토할 수 있게 파일로 남긴다."""
    cnt: Counter[str] = Counter()
    pages: dict[str, set[int]] = defaultdict(set)
    for i, text in enumerate(texts):
        for line in text.split("\n"):
            for kind, w, _, _ in util.iter_words(line):
                if kind == "ko":
                    stem, _ = util.split_josa(w)
                    w = stem
                    if len(w) < 2:
                        continue
                elif kind == "en":
                    if len(w) < 3:
                        continue
                else:
                    continue
                cnt[w] += 1
                pages[w].add(i)
    return sorted(((w, c, len(pages[w])) for w, c in cnt.items() if c >= min_count and len(pages[w]) >= min_pages),
                  key=lambda x: (-x[1], x[0]))


def write_glossary(path: Path, items: list[tuple[str, int, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ["# 책에서 자동 수집한 빈도 높은 용어 (용어<TAB>횟수<TAB>쪽수).",
            "# 같은 오인식이 반복돼 섞였을 수 있으니 틀린 항목은 지우세요. 이 목록의 용어는 자동 수정에서 보호됩니다."]
    path.write_text("\n".join(head + [f"{w}\t{c}\t{p}" for w, c, p in items]) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ 줄 단위 규칙
_NUM_RUN = re.compile(r"[0-9OolI|](?:[0-9OolI|,.]*[0-9OolI|])?")
_RN_UNIT = re.compile(r"(?<=\d)((?:rn)+)(?![A-Za-z])")
_SPACE_BEFORE_PUNCT = re.compile(r"(?<=[A-Za-z가-힣])[ \t]+(?=[,.;:!?)\]](?:\s|$|[가-힣]))")
_SPACE_INSIDE_PAREN = re.compile(r"\([ \t]+|[ \t]+\)")
_MULTI_SPACE = re.compile(r"(?<=\S)[ \t]{2,}")
_FIG = util.FIG_MARK


def _fix_numeric_runs(token: str) -> str:
    """토큰 안의 숫자열 중 '숫자 문맥'이 확실한 것만 O/l/I → 0/1 로 고친다."""
    def repl(m: re.Match) -> str:
        run = m.group(0)
        s, e = m.span()
        real = sum(c.isdigit() for c in run)
        conf = sum(c in "OolI|" for c in run)
        if real == 0 or conf == 0:
            return run
        before = token[s - 1] if s > 0 else ""
        after = token[e] if e < len(token) else ""
        if before.isascii() and before.isalpha():       # 'RSI14' 의 I14 처럼 영단어 안쪽: 건드리지 않음
            return run
        if after.isascii() and after.isalpha():
            return run
        unit = after in UNIT_FOLLOW
        ok = (conf <= real) if unit else (real >= 2 and conf * 2 <= real)
        return run.translate(DIGIT_FIX) if ok else run
    return _NUM_RUN.sub(repl, token)


def normalize_line(line: str, protect: Protect | None = None, quote_style: str = "straight") -> tuple[str, list[str]]:
    """한 줄을 정규화. 반환: (새 줄, 적용된 규칙 id 목록)."""
    protect = protect or Protect()
    rules: list[str] = []

    def note(rule: str, new: str, old: str) -> str:
        if new != old and rule not in rules:
            rules.append(rule)
        return new

    out = line
    # 그림 표식은 그대로 보존
    marks: list[str] = []

    def stash(m: re.Match) -> str:
        marks.append(m.group(0))
        return f"{len(marks) - 1}"

    out = _FIG.sub(stash, out)

    o = out
    out = unicodedata.normalize("NFC", out).translate(ZERO_WIDTH).translate(SPACE_VARIANTS)
    note("nfc", out, o)

    o = out
    out = "".join(unicodedata.normalize("NFKC", c) if _FULLWIDTH_OK(c) else c for c in out)
    note("fullwidth", out, o)

    if quote_style == "straight":
        o = out
        out = out.translate(QUOTE_MAP)
        note("quote", out, o)

    o = out
    out = out.translate(HYPHEN_MAP)
    note("hyphen", out, o)

    # 공백 정리 (들여쓰기·표 구분 구조는 유지)
    o = out
    lead = len(out) - len(out.lstrip(" \t"))
    head, body = out[:lead].replace("\t", "    "), out[lead:]
    body = body.rstrip()
    body = _MULTI_SPACE.sub(" ", body.replace("\t", " "))
    if not body.lstrip().startswith("|"):
        body = _SPACE_BEFORE_PUNCT.sub("", body)
        body = _SPACE_INSIDE_PAREN.sub(lambda m: "(" if m.group(0).startswith("(") else ")", body)
    out = head + body if body else ""
    note("space", out, o)

    # 숫자 문맥 혼동 (보호 토큰 제외)
    o = out
    pieces = re.split(r"(\s+)", out)
    for i, t in enumerate(pieces):
        if not t or t.isspace() or "" in t:
            continue
        core = t.strip("\"'()[]{},.;:!?")
        if core in protect or t in protect:
            continue
        t2 = _fix_numeric_runs(t)
        t2 = _RN_UNIT.sub(lambda m: "m" * (len(m.group(1)) // 2), t2)
        pieces[i] = t2
    out = "".join(pieces)
    note("digit_conf", out, o)

    out = re.sub("(\\d+)", lambda m: marks[int(m.group(1))], out)
    return out, rules


@dataclass
class NormResult:
    text: str
    changes: list[dict] = field(default_factory=list)     # {line, before, after, rules}


def normalize_text(md: str, protect: Protect | None = None, quote_style: str = "straight") -> NormResult:
    """쪽 Markdown 전체를 줄 단위로 정규화. 줄 수는 바뀌지 않는다."""
    lines = util.lines_of(md)
    out, changes = [], []
    for i, ln in enumerate(lines, 1):
        new, rules = normalize_line(ln, protect, quote_style)
        out.append(new)
        if new != ln:
            changes.append({"line": i, "before": ln, "after": new, "rules": rules})
    return NormResult("\n".join(out), changes)


def revert_text(normalized: str, changes: list[dict]) -> str:
    """normalize_text 의 changes 로 원문을 복원한다(줄 단위 이전 값으로 되돌림)."""
    lines = util.lines_of(normalized)
    for c in changes:
        i = c["line"] - 1
        if i >= len(lines) or lines[i] != c["after"]:
            raise ValueError(f"되돌릴 수 없음: {c['line']}행이 기록된 '이후' 값과 다릅니다.")
        lines[i] = c["before"]
    return "\n".join(lines)
