"""M2 탐지: 의심 구간만 추린다. 고치지는 않는다(후보·수정은 M3~M5).

신호(name):
  low_conf         (a) Tesseract 단어 신뢰도(tsv)가 기준 미만
  ocr_disagree     (b) 두 OCR(Tesseract vs Gemini) 결과를 단어 정렬했을 때 서로 다름
  rare_near_freq   (c) 책 안에서 드문 낱말인데 자모 편집거리 1~2 안에 훨씬 흔한 낱말이 있음
  year_range       (d) '○○○○년' 이 허용 연도 범위 밖이거나 자릿수가 이상함
  caption_seq      (d) 그림/표 캡션 번호가 앞뒤 번호 흐름에서 혼자 벗어남
  sum_mismatch     (d) 표의 합계 열/행이 위 항목의 합과 다름
  english_far      (e) 영문 토큰이 약어 화이트리스트·용어집·책 안 흔한 영단어와 가까운데 같지 않음(R5I→RSI) / (선택) 사전에 없음
  mixed_script     한글 사이에 영문·숫자 한 글자가 낀 낱말(한l글)
  syllable_split   한글 한 음절짜리 토큰이 3개 이상 연달아(글자 사이 공백)
  unreadable       [판독불가] 표식
  bad_char         깨진 글자(�, □, 홀로 있는 자모)

출력: span = {id, page, line, col, end, token, kind, signals[{name,weight,detail}], score, box, line_box, alt, suggest}
  score = 1 - Π(1 - weight). 단어 상자(box)는 tesseract tsv 가 있을 때만 채워진다.
쪽번호 연속성은 본문에 글자가 없으므로(쪽번호는 OCR 단계에서 이미 뺌) 구간이 아니라 쪽 단위 노트(page_notes)로 따로 낸다.
"""
from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import util
from .candidates import Vocab
from .normalize import Protect, normalize_line

SIGNAL_CLASS_NUMERIC_OK = {"low_conf", "ocr_disagree", "year_range", "caption_seq", "sum_mismatch",
                           "english_far", "mixed_script", "unreadable", "bad_char"}


@dataclass
class Params:
    conf_thr: float = 60.0              # tsv 단어 신뢰도(0~100) 미만이면 low_conf
    rare_max: int = 1                   # 이 횟수 이하로 나온 낱말만 rare 로 봄
    min_score: float = 0.30             # 이 점수 미만 구간은 버림
    year_min: int = 1900
    year_max: int = 2035
    english_dict: set[str] | None = None   # 선택: 영어 단어 목록(소문자). 없으면 '사전에 없음' 신호는 쓰지 않음
    max_alt_span: int = 3               # 두 OCR 정렬에서 이 개수 이하의 조각 차이만 신호로 사용


def noisy_or(ws: list[float]) -> float:
    p = 1.0
    for w in ws:
        p *= 1.0 - w
    return round(1.0 - p, 3)


def span_id(page: int, line: int, col: int) -> str:
    return f"p{page:04d}-L{line:03d}-C{col:03d}"


class _Spans:
    """같은 (쪽, 줄, 시작, 끝) 의 신호를 하나의 span 으로 합친다."""

    def __init__(self):
        self.by_key: dict[tuple, dict] = {}

    def add(self, page: int, line: int, col: int, end: int, token: str, kind: str, name: str, weight: float,
            detail: str = "", **extra) -> dict:
        key = (page, line, col, end)
        sp = self.by_key.get(key)
        if sp is None:
            sp = {"id": span_id(page, line, col), "page": page, "line": line, "col": col, "end": end, "token": token,
                  "kind": kind, "signals": [], "score": 0.0, "box": None, "line_box": None, "alt": None, "suggest": []}
            self.by_key[key] = sp
        if not any(s["name"] == name for s in sp["signals"]):
            sp["signals"].append({"name": name, "weight": round(weight, 3), "detail": detail})
        for k, v in extra.items():
            if k == "suggest":
                for sg in v:
                    if sg not in sp["suggest"]:
                        sp["suggest"].append(sg)
            elif v is not None and not sp.get(k):
                sp[k] = v
        return sp

    def result(self, min_score: float) -> list[dict]:
        out = []
        for sp in self.by_key.values():
            sp["score"] = noisy_or([s["weight"] for s in sp["signals"]])
            if sp["score"] >= min_score:
                out.append(sp)
        out.sort(key=lambda s: (s["page"], s["line"], s["col"]))
        return out


# ------------------------------------------------------------------ tesseract tsv
def parse_word_tsv(tsv: str) -> list[dict]:
    """tesseract tsv(level 5 행) → [{text, conf, box:[x0,y0,x1,y1], key:(block,par,line)}] 읽는 순서."""
    out = []
    for row in tsv.splitlines()[1:]:
        c = row.split("\t")
        if len(c) < 12 or c[0] != "5":
            continue
        text = c[11].strip()
        try:
            conf = float(c[10])
            x, y, w, h = int(c[6]), int(c[7]), int(c[8]), int(c[9])
        except ValueError:
            continue
        if not text or conf < 0:
            continue
        out.append({"text": text, "conf": conf, "box": [x, y, x + w, y + h], "key": (int(c[2]), int(c[3]), int(c[4]))})
    return out


def load_tsv(page: dict, cache_dir: Path, run: bool = True, lang: str = "kor+eng") -> list[dict] | None:
    """캐시된 tsv 가 있으면 읽고, 없으면(tesseract 가 있고 run=True 일 때) 쪽 이미지로 만든다. 불가능하면 None."""
    cache = Path(cache_dir) / f"p{page['page']:04d}.tsv"
    if cache.exists():
        return parse_word_tsv(cache.read_text(encoding="utf-8"))
    if not run or not page.get("image") or not Path(page["image"]).exists():
        return None
    from .. import ocr_tesseract
    exe = ocr_tesseract.find_binary()
    if not exe:
        return None
    import os
    import subprocess
    proc = subprocess.run([exe, str(page["image"]), "stdout", "-l", lang, "--psm", "3", "--dpi", "300", "tsv"],
                          capture_output=True, timeout=240, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    if proc.returncode != 0:
        return None
    text = proc.stdout.decode("utf-8", "replace")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(text, encoding="utf-8")
    return parse_word_tsv(text)


def attach_tsv(lines: list[str], words: list[dict]) -> dict[tuple[int, int], dict]:
    """본문 공백 토큰 ↔ tsv 단어를 순서 정렬로 짝지어 {(줄, 토큰시작): {conf, box, line_box}} 를 돌려준다."""
    md_toks = []
    for li, ln in enumerate(lines, 1):
        for t in util.tokenize_line(ln):
            md_toks.append((li, t))
    a = [t.text for _, t in md_toks]
    b = [normalize_line(w["text"])[0] for w in words]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out: dict[tuple[int, int], dict] = {}
    line_boxes: dict[int, list[int]] = {}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            li, t = md_toks[i1 + k]
            w = words[j1 + k]
            out[(li, t.start)] = {"conf": w["conf"], "box": w["box"]}
            lb = line_boxes.get(li)
            line_boxes[li] = list(w["box"]) if lb is None else [min(lb[0], w["box"][0]), min(lb[1], w["box"][1]),
                                                                  max(lb[2], w["box"][2]), max(lb[3], w["box"][3])]
    for (li, _), rec in out.items():
        rec["line_box"] = line_boxes.get(li)
    return out


# ------------------------------------------------------------------ 개별 신호
_CAPTION = re.compile(r"^\s*(그림|표|도표|Figure|Fig\.?|Table)\s*(\d+)(?:\s*[-.]\s*(\d+))?")
_YEAR = re.compile(r"(?<![\d,.])(\d[\d,]*)(\s*)년")
_TOTAL_LABELS = {"합계", "총계", "계", "Total", "total", "합 계", "총 계"}
_NUM_CELL = re.compile(r"^\s*[-+]?\d[\d,]*(?:\.\d+)?\s*%?\s*$")


def _num(s: str) -> float:
    return float(s.strip().rstrip("%").replace(",", "").strip())


def sig_year(spans: _Spans, page: int, li: int, line: str, p: Params) -> None:
    for m in _YEAR.finditer(line):
        digits = m.group(1).replace(",", "")
        n = len(digits)
        if n <= 2:
            continue                                  # '3년', '21년' 은 기간일 수 있음
        if n == 4 and p.year_min <= int(digits) <= p.year_max:
            continue
        why = (f"{digits}년: 자릿수 {n}" if n != 4 else f"{digits}년: 허용 범위 {p.year_min}~{p.year_max} 밖")
        spans.add(page, li, m.start(1), m.end(1), m.group(1), "num", "year_range", 0.6, why)


def sig_caption(spans: _Spans, page_lines: list[tuple[int, int, str]], p: Params) -> None:
    """(쪽, 줄번호, 줄) 목록을 책 순서로 받아, 종류(그림/표)별 캡션 번호 흐름에서 '혼자' 벗어난 번호를 찾는다.

    번호는 (장, 순번) 쌍('1-3' → (1,3), '3' → (0,3)). 앞 번호와 같거나 +1(또는 새 장의 1번)이면 이어지는 것으로 본다.
    앞뒤 번호끼리는 서로 이어지는데 이 번호만 앞뒤 어느 쪽에도 안 맞으면 이 번호를 의심한다(번호 체계가 통째로 바뀐 경우는 건드리지 않음).
    """
    kind_of = {"Figure": "fig", "Fig.": "fig", "Fig": "fig", "그림": "fig", "도표": "fig", "표": "tab", "Table": "tab"}
    series: dict[str, list[tuple]] = {}
    for pg, li, ln in page_lines:
        m = _CAPTION.match(ln)
        if not m:
            continue
        has_ch = m.group(3) is not None
        num = (int(m.group(2)), int(m.group(3))) if has_ch else (0, int(m.group(2)))
        end = m.end(3) if has_ch else m.end(2)
        series.setdefault(kind_of[m.group(1)], []).append((pg, li, num, m.start(2), end, ln[m.start(2):end], has_ch))

    def fits(a, b):                       # b 가 a 다음(또는 같은) 번호인가
        return (b[0] == a[0] and b[1] in (a[1], a[1] + 1)) or (b[0] == a[0] + 1 and b[1] == 1)

    def label(n, has_ch):
        return f"{n[0]}-{n[1]}" if has_ch else str(n[1])

    for items in series.values():
        n = len(items)
        for i, (pg, li, cur, cs, ce, txt, has_ch) in enumerate(items):
            prev = items[i - 1][2] if i > 0 else None
            nxt = items[i + 1][2] if i + 1 < n else None
            sug = None
            if prev is not None and nxt is not None:
                if fits(prev, cur) or fits(cur, nxt):
                    continue
                gap_ok = (nxt[0] == prev[0] and nxt[1] in (prev[1] + 1, prev[1] + 2)) or fits(prev, nxt)
                if not gap_ok:
                    continue
                if nxt[0] == prev[0] and nxt[1] == prev[1] + 2:
                    sug = label((prev[0], prev[1] + 1), has_ch)
            elif prev is not None and nxt is None and n >= 3:       # 마지막 번호
                pp = items[i - 2][2]
                if fits(prev, cur) or not fits(pp, prev):
                    continue
                sug = label((prev[0], prev[1] + 1), has_ch)
            elif prev is None and nxt is not None and n >= 3:       # 첫 번호
                n2 = items[i + 2][2]
                if fits(cur, nxt) or not fits(nxt, n2):
                    continue
                sug = label((nxt[0], nxt[1] - 1), has_ch) if nxt[1] > 1 else None
            else:
                continue
            spans.add(pg, li, cs, ce, txt, "label", "caption_seq", 0.55,
                      f"캡션 번호 {txt}: 앞 {prev} 뒤 {nxt} 흐름에서 벗어남",
                      suggest=[{"text": sug, "source": "sequence"}] if sug else [])


def sig_table_sums(spans: _Spans, page: int, lines: list[str]) -> None:
    i = 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        block = list(range(i, j))
        i = j
        rows = []
        for k in block:
            if re.match(r"^\s*\|[\s:|-]+\|?\s*$", lines[k]):
                continue
            cells = []
            pos = lines[k].index("|") + 1
            for cell in lines[k].strip().strip("|").split("|"):
                lead = len(cell) - len(cell.lstrip())
                cells.append((cell.strip(), pos + lead))
                pos += len(cell) + 1
            rows.append((k, cells))
        tot = [r for r in rows if r[1] and r[1][0][0].replace(" ", "") in {t.replace(" ", "") for t in _TOTAL_LABELS}]
        if len(rows) < 4 or not tot:
            continue
        trow = tot[0]
        body = [r for r in rows[1:] if r is not trow and r[0] < trow[0]]
        for ci in range(1, len(trow[1])):
            cells = [r[1][ci] for r in body if ci < len(r[1])]
            if len(cells) < 2 or ci >= len(trow[1]) or not all(_NUM_CELL.match(c[0]) for c in cells) \
                    or not _NUM_CELL.match(trow[1][ci][0]):
                continue
            total = _num(trow[1][ci][0])
            ssum = sum(_num(c[0]) for c in cells)
            if abs(ssum - total) <= max(0.5, 0.005 * abs(total)):
                continue
            detail = f"열 {ci + 1}: 위 항목 합 {ssum:g} ≠ 합계 {total:g}"
            tk = trow[0]
            tc = trow[1][ci]
            spans.add(page, tk + 1, tc[1], tc[1] + len(tc[0]), tc[0], "num", "sum_mismatch", 0.45, detail)
            for r in body:
                if ci < len(r[1]):
                    c = r[1][ci]
                    spans.add(page, r[0] + 1, c[1], c[1] + len(c[0]), c[0], "num", "sum_mismatch", 0.3, detail)


_BAD = re.compile("[�□◻]")
_LONE_JAMO = re.compile(r"(?<![ㄱ-ㅎㅏ-ㅣ])[ㄱ-ㅎㅏ-ㅣ](?![ㄱ-ㅎㅏ-ㅣ])")
_MIXED = re.compile(r"[가-힣][A-Za-z0-9|][가-힣]|[가-힣][lI|]$|^[lI|][가-힣]")
_LEGIT_ORDINAL = re.compile(r"제\d+")        # 제1장, 제2차 …


def sig_chars(spans: _Spans, page: int, li: int, line: str) -> None:
    pos = 0
    toks = util.tokenize_line(line)
    for t in toks:
        core = t.core
        if not core:
            continue
        if core == "[판독불가]" or util.UNREADABLE in t.text:
            k = t.text.index(util.UNREADABLE) + t.start
            spans.add(page, li, k, k + len(util.UNREADABLE), util.UNREADABLE, "unreadable", "unreadable", 0.9, "OCR 이 읽지 못한 부분")
            continue
        if util.FIG_MARK.search(t.text):
            continue
        for m in _BAD.finditer(t.text):
            spans.add(page, li, t.start + m.start(), t.start + m.end(), m.group(0), "char", "bad_char", 0.6, "깨진 글자")
        for m in _LONE_JAMO.finditer(t.text):
            spans.add(page, li, t.start + m.start(), t.start + m.end(), m.group(0), "char", "bad_char", 0.5, "홀로 있는 자모")
        if util.has_hangul(core) and _MIXED.search(_LEGIT_ORDINAL.sub("제", core)):
            spans.add(page, li, t.cstart, t.cend, core, "mixed", "mixed_script", 0.5, "한글 사이에 영문/숫자 한 글자")
    # 음절 분리: 한 음절 토큰이 3개 이상 연속
    run = []
    for t in toks + [None]:
        if t is not None and len(t.core) == 1 and util.is_hangul_syllable(t.core) and t.text == t.core:
            run.append(t)
            continue
        if len(run) >= 3:
            s, e = run[0].start, run[-1].end
            joined = "".join(r.core for r in run)
            spans.add(page, li, s, e, line[s:e], "split", "syllable_split", 0.5,
                      f"한 음절 토큰 {len(run)}개 연속", suggest=[{"text": joined, "source": "join"}])
        run = []


# ------------------------------------------------------------------ 두 OCR 비교
def _frags(lines: list[str]) -> list[tuple[str, int, int, int, str]]:
    out = []
    for li, ln in enumerate(lines, 1):
        for kind, w, s, e in util.iter_words(ln):
            out.append((w, li, s, e, kind))
    return out


def sig_alt(spans: _Spans, page: int, lines: list[str], alt_lines: list[str], p: Params) -> None:
    a, b = _frags(lines), _frags(alt_lines)
    sm = difflib.SequenceMatcher(None, [x[0] for x in a], [x[0] for x in b], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        n, m = i2 - i1, j2 - j1
        if n > p.max_alt_span or m > p.max_alt_span:
            continue
        if n == m:
            for k in range(n):
                w, li, s, e, kind = a[i1 + k]
                alt = b[j1 + k][0]
                num = kind == "num" or util.has_digit(w)
                spans.add(page, li, s, e, w, kind, "ocr_disagree", 0.75 if num else 0.6,
                          f"다른 OCR: '{alt}'", alt=alt)
        else:
            lis = {a[x][1] for x in range(i1, i2)}
            if len(lis) != 1:
                continue
            li = lis.pop()
            s, e = a[i1][2], a[i2 - 1][3]
            alt = " ".join(x[0] for x in b[j1:j2])
            spans.add(page, li, s, e, lines[li - 1][s:e], "split" if n > 1 else a[i1][4], "ocr_disagree", 0.55,
                      f"다른 OCR: '{alt}'", alt=alt)


# ------------------------------------------------------------------ 책 안 빈도·영문
def sig_vocab(spans: _Spans, page: int, li: int, line: str, vocab: Vocab, protect: Protect, p: Params) -> None:
    for kind, w, s, e in util.iter_words(line):
        if w in protect or (kind == "ko" and util.split_josa(w)[0] in protect):
            continue
        if kind == "ko":
            if len(w) < 2 or vocab.freq_ko(w) > p.rare_max:
                continue
            cands = vocab.lookup_ko(w)
            if not cands:
                continue
            best = cands[0]
            base = 0.55 if best["wdist"] <= 0.5 else 0.45 if best["wdist"] <= 1.0 else 0.30
            wt = base * (0.7 + 0.3 * min(1.0, math.log(max(best["freq"], 1)) / math.log(10)))
            spans.add(page, li, s, e, w, "ko", "rare_near_freq", wt,
                      f"드문 낱말(책 안 {vocab.freq_ko(w)}회) ↔ '{best['text']}'({best['freq']}회, 자모거리 {best['wdist']})")
        elif kind == "en" and len(w) >= 3:
            if vocab.freq_en(w) > p.rare_max and w not in protect:
                continue
            cands = vocab.lookup_en(w)
            if cands:
                best = cands[0]
                wt = 0.5 if best["dist"] <= 1 else 0.35
                spans.add(page, li, s, e, w, "en", "english_far", wt,
                          f"영문 '{w}' ↔ '{best['text']}'({best['source']}, 거리 {best['dist']})")
            elif p.english_dict is not None and w.lower() not in p.english_dict and not util.has_digit(w):
                spans.add(page, li, s, e, w, "en", "english_far", 0.3, f"영어 사전에 없음: '{w}'")


# ------------------------------------------------------------------ 쪽번호 연속성(구간이 아닌 쪽 노트)
def page_number_notes(pages: list[dict]) -> list[dict]:
    """인쇄 쪽번호가 (쪽 순번 차이만큼) 증가하는 흐름에서 혼자 벗어난 쪽. 본문에 글자가 없어 수정 대상이 아니라 노트로만 남긴다."""
    seq = [(p["page"], p["page_number"]) for p in pages if isinstance(p.get("page_number"), int)]
    if len(seq) < 3:
        return []
    notes = []
    for i, (pg, pn) in enumerate(seq):
        offsets = [pn - pg]
        near = [q for q in seq if q[0] != pg and abs(q[0] - pg) <= 3]
        if not near:
            continue
        votes = [(qn - qg) for qg, qn in near]
        top = max(set(votes), key=votes.count)
        if votes.count(top) >= 2 and offsets[0] != top:
            notes.append({"page": pg, "kind": "pageno", "read": pn, "expected": pg + top,
                          "detail": f"인쇄 쪽번호 {pn} 읽음, 주변 흐름상 {pg + top} 예상"})
    return notes


# ------------------------------------------------------------------ 진입점
def detect_pages(pages: list[dict], vocab: Vocab, protect: Protect, params: Params | None = None,
                 alt_texts: dict[int, str] | None = None, tsv: dict[int, list[dict]] | None = None) -> dict:
    """정규화된 쪽(pages[i]['text'])에서 의심 구간을 찾는다.

    pages: [{page, text, page_number?}]   alt_texts: {쪽: 다른 OCR 의 정규화 텍스트}   tsv: {쪽: parse_word_tsv 결과}
    반환: {spans, page_notes, stats}
    """
    p = params or Params()
    sp = _Spans()
    cap_lines: list[tuple[int, int, str]] = []
    for pg in pages:
        n = pg["page"]
        lines = util.lines_of(pg["text"])
        for li, ln in enumerate(lines, 1):
            if not ln.strip() or util.FIG_MARK.fullmatch(ln.strip()) or re.match(r"^\s*\|[\s:|-]+\|?\s*$", ln):
                continue
            sig_chars(sp, n, li, ln)
            sig_year(sp, n, li, ln, p)
            sig_vocab(sp, n, li, ln, vocab, protect, p)
            cap_lines.append((n, li, ln))
        sig_table_sums(sp, n, lines)
        if alt_texts and n in alt_texts:
            sig_alt(sp, n, lines, util.lines_of(alt_texts[n]), p)
    sig_caption(sp, cap_lines, p)
    # tsv: 신뢰도 신호 + 단어/줄 상자
    for pg in pages:
        n = pg["page"]
        words = (tsv or {}).get(n)
        if not words:
            continue
        lines = util.lines_of(pg["text"])
        att = attach_tsv(lines, words)
        for (li, start), rec in att.items():
            if rec["conf"] < p.conf_thr and lines[li - 1][start:start + 1].strip():
                t = next((t for t in util.tokenize_line(lines[li - 1]) if t.start == start), None)
                if t is None or not t.core or util.FIG_MARK.search(t.text):
                    continue
                if t.core in protect:
                    continue
                wt = min(0.75, max(0.25, 0.25 + 0.5 * (1 - rec["conf"] / p.conf_thr)))
                kind = "ko" if util.has_hangul(t.core) else ("num" if util.has_digit(t.core) and not util.has_latin(t.core) else "en")
                sp.add(n, li, t.cstart, t.cend, t.core, kind, "low_conf", wt, f"tesseract 단어 신뢰도 {rec['conf']:.0f}")
        # 상자 채우기: span 이 걸치는 공백 토큰의 상자
        for s in sp.by_key.values():
            if s["page"] != n:
                continue
            for t in util.tokenize_line(lines[s["line"] - 1]):
                if t.start <= s["col"] < t.end or t.start < s["end"] <= t.end:
                    rec = att.get((s["line"], t.start))
                    if rec:
                        s["box"] = s["box"] or rec["box"]
                        s["line_box"] = rec["line_box"]
                        break
    spans = sp.result(p.min_score)
    # 같은 위치의 span 에 줄 본문을 붙여 두면 리포트가 쉬워진다
    text_by_page = {pg["page"]: util.lines_of(pg["text"]) for pg in pages}
    for s in spans:
        s["context"] = text_by_page[s["page"]][s["line"] - 1]
    stats: dict[str, int] = {}
    for s in spans:
        for sg in s["signals"]:
            stats[sg["name"]] = stats.get(sg["name"], 0) + 1
    return {"spans": spans, "page_notes": page_number_notes(pages),
            "stats": {"spans": len(spans), "by_signal": dict(sorted(stats.items())),
                      "pages": len(pages), "tsv_pages": sum(1 for pg in pages if (tsv or {}).get(pg["page"])),
                      "alt_pages": sum(1 for pg in pages if alt_texts and pg["page"] in alt_texts)}}
