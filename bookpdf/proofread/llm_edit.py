"""M4 LLM 수정 목록: 의심 구간(전후 문맥 포함)만 Gemini 에 보내 수정 목록(원문→수정, 이유, 확신도)을 받고, 코드로 엄격히 검증한다.

원칙: LLM 은 '제안'만 한다. 아래 검사를 통과하지 못한 제안은 자동 거절하고 사유를 기록한다.
  · 형식 위반(JSON/필드/자료형), 알 수 없는 id
  · 원문 불일치(줄에 없는 문자열을 '원문'이라고 주장 = 환각) / 의심 구간 밖 수정
  · 편집거리·길이 변화 과다(글자 편집거리, 자모 편집거리, 길이 차, 비율)
  · 숫자·영문 변경인데 해당 구간에 숫자/영문 관련 탐지 신호가 없음
  · 보호 용어(용어집·약어) 수정, 같은 줄 겹치는 수정, 줄당 수정 수 초과
확신도가 기준 미만이면 거절이 아니라 '보류(held)'. temperature 0 은 호출기에서 고정한다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from . import util
from .cache import ResponseCache, key_of
from .detect import SIGNAL_CLASS_NUMERIC_OK
from .normalize import Protect

SCHEMA = {
    "type": "object",
    "properties": {"edits": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "original": {"type": "string"}, "corrected": {"type": "string"},
                       "reason": {"type": "string"}, "confidence": {"type": "number"}},
        "required": ["id", "original", "corrected", "confidence"]}}},
    "required": ["edits"],
}

PROMPT = """당신은 한국어 책(스캔 OCR 결과)의 교정 보조입니다. 아래 '검토 항목'의 target 만 검토합니다.
규칙(반드시 지키세요):
1. 수정할 수 있는 것은 각 target 토큰뿐입니다. line/before/after 는 문맥 참고용이며 절대 고치지 마세요.
2. OCR 이 글자를 잘못 읽은 경우만 고칩니다. 맞춤법·띄어쓰기·문체·표현 개선, 새 내용 추가, 문장 바꾸기는 금지입니다.
3. 확실하지 않으면 수정 목록에 넣지 마세요(빈 목록이 정답일 수 있습니다). 없는 사실을 지어내지 마세요.
4. 숫자와 영문은 해당 항목의 signals 에 숫자/영문과 관련된 근거가 있을 때만 고치세요. 후보(candidates)는 참고일 뿐이며 맞다고 보장되지 않습니다.
5. original 에는 line 에 실제로 있는 그대로의 문자열을, corrected 에는 고친 문자열을 쓰세요. confidence 는 0~1 로 정직하게.
6. 출력은 JSON 만: {{"edits":[{{"id":"항목 id","original":"...","corrected":"...","reason":"한 줄 이유","confidence":0.0}}]}}

검토 항목(JSON):
{items}
"""


@dataclass
class Limits:
    max_char_dist: int = 3          # 원문→수정 글자 편집거리 상한
    max_jamo_dist: int = 4          # 자모 편집거리 상한
    max_len_delta: int = 2          # 길이 변화 상한(글자)
    max_edit_ratio: float = 0.5     # 편집거리 / 원문 길이 상한
    min_confidence: float = 0.7     # 미만이면 거절이 아니라 보류
    max_edits_per_line: int = 3
    window: int = 3                 # 의심 구간 양옆으로 허용하는 여유 글자 수(조사 등)
    batch_lines: int = 8            # 요청 하나에 담는 줄 수


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def _latin(s: str) -> str:
    return "".join(c for c in s if c.isascii() and c.isalpha())


def locate(line: str, original: str, span: dict, window: int) -> tuple[int, int] | None:
    """줄에서 original 의 위치를 찾는다. 의심 구간과 겹치는 위치를 우선하며, 겹치는 곳이 없으면 None."""
    if not original:
        return None
    best = None
    start = 0
    while True:
        i = line.find(original, start)
        if i < 0:
            break
        j = i + len(original)
        if i < span["end"] + window and j > span["col"] - window and i < span["end"] and j > span["col"]:
            d = abs(i - span["col"]) + abs(j - span["end"])
            if best is None or d < best[0]:
                best = (d, i, j)
        start = i + 1
    return (best[1], best[2]) if best else None


def validate(span: dict, line: str, raw: dict, lim: Limits, protect: Protect) -> dict:
    """제안 하나를 검증. 반환: {status: proposed|held|rejected|nochange, code, detail, start, end, original, corrected, ...}"""
    base = {"id": span["id"], "page": span["page"], "line": span["line"], "source": "llm"}

    def rej(code: str, detail: str, **kw) -> dict:
        return {**base, "status": "rejected", "code": code, "detail": detail,
                "original": raw.get("original"), "corrected": raw.get("corrected"), **kw}

    orig, corr = raw.get("original"), raw.get("corrected")
    if not isinstance(orig, str) or not isinstance(corr, str):
        return rej("형식", "original/corrected 가 문자열이 아님")
    try:
        conf = float(raw.get("confidence"))
    except (TypeError, ValueError):
        return rej("형식", "confidence 가 숫자가 아님")
    if not 0.0 <= conf <= 1.0:
        return rej("형식", f"confidence 범위 밖({conf})")
    if "\n" in corr or "\r" in corr or "\x00" in corr:
        return rej("형식", "수정문에 줄바꿈/제어문자")
    if orig == corr:
        return {**base, "status": "nochange", "code": "변경없음", "detail": "", "original": orig, "corrected": corr}
    if not orig.strip():
        return rej("형식", "원문이 비어 있음")
    pos = locate(line, orig, span, lim.window)
    if pos is None:
        code = "원문_불일치" if orig not in line else "구간_밖"
        return rej(code, "줄에 없는 문자열을 원문이라 주장(환각 가능)" if code == "원문_불일치"
                   else f"의심 구간({span['token']}) 밖의 텍스트를 수정하려 함")
    s, e = pos
    names = {x["name"] for x in span["signals"]}
    only_space = orig.replace(" ", "") == corr.replace(" ", "")
    d_c = util.levenshtein(orig, corr)
    d_j = util.jamo_distance(orig, corr)
    if only_space:
        if not names & {"syllable_split", "ocr_disagree"}:
            return rej("띄어쓰기_신호없음", "띄어쓰기만 바꾸는 수정인데 음절분리/OCR불일치 신호가 없음", start=s, end=e)
        if abs(len(orig) - len(corr)) > 12:
            return rej("과다변경", "띄어쓰기 변경이 너무 많음", start=s, end=e)
    else:
        if abs(len(orig) - len(corr)) > lim.max_len_delta:
            return rej("길이변화", f"길이 차 {abs(len(orig) - len(corr))} > {lim.max_len_delta}", start=s, end=e)
        if d_c > lim.max_char_dist or d_j > lim.max_jamo_dist or d_c / max(len(orig), 1) > lim.max_edit_ratio:
            return rej("편집거리", f"글자거리 {d_c}, 자모거리 {d_j}, 비율 {d_c / max(len(orig), 1):.2f}", start=s, end=e)
        if _digits(orig) != _digits(corr) or _latin(orig) != _latin(corr):
            if not names & SIGNAL_CLASS_NUMERIC_OK:
                return rej("숫자영문_신호없음", f"숫자/영문을 바꾸는데 구간 신호가 {sorted(names)} 뿐", start=s, end=e)
    stem = util.split_josa(orig.strip(".,;:!?()\"'"))[0]
    if orig.strip() in protect or stem in protect:
        return rej("보호어", "용어집/약어에 있는 낱말", start=s, end=e)
    out = {**base, "status": "proposed" if conf >= lim.min_confidence else "held",
           "code": "" if conf >= lim.min_confidence else "낮은확신", "detail": "" if conf >= lim.min_confidence else f"확신도 {conf:.2f} < {lim.min_confidence}",
           "original": orig, "corrected": corr, "start": s, "end": e, "reason": str(raw.get("reason") or "")[:200],
           "confidence": round(conf, 3), "char_dist": d_c, "jamo_dist": d_j, "numeric": _digits(orig) != _digits(corr) or _latin(orig) != _latin(corr)}
    return out


def _parse(text: str) -> list[dict]:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(t)
    if not isinstance(data, dict) or not isinstance(data.get("edits"), list):
        raise ValueError("응답에 edits 배열이 없습니다.")
    for e in data["edits"]:
        if not isinstance(e, dict):
            raise ValueError("edits 항목이 객체가 아닙니다.")
    return data["edits"]


def build_items(batch: list[tuple[int, list[dict]]], texts: dict[int, list[str]], cands: dict[str, list[dict]]) -> list[dict]:
    items = []
    for page, spans in batch:
        lines = texts[page]
        li = spans[0]["line"]
        items.append({
            "line_id": f"p{page:04d}-L{li:03d}", "before": lines[li - 2] if li >= 2 else "", "line": lines[li - 1],
            "after": lines[li] if li < len(lines) else "",
            "targets": [{"id": s["id"], "target": s["token"],
                         "signals": [f"{g['name']}: {g['detail']}" if g["detail"] else g["name"] for g in s["signals"]],
                         "candidates": [c["text"] for c in cands.get(s["id"], [])]} for s in spans]})
    return items


@dataclass
class LlmStats:
    requests: int = 0
    cache_hits: int = 0
    format_retries: int = 0
    proposed: int = 0
    held: int = 0
    rejected: int = 0
    nochange: int = 0
    unmatched_targets: int = 0      # 응답에 아무 제안도 없던 의심 구간 수(정상: 모델이 수정하지 않기로 함)
    errors: list[str] = field(default_factory=list)
    stopped: str | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def propose_edits(spans: list[dict], texts: dict[int, list[str]], cands: dict[str, list[dict]], caller: Callable[..., str],
                  protect: Protect, lim: Limits | None = None, cache: ResponseCache | None = None,
                  max_format_retries: int = 2, log: Callable[[str], None] = print) -> tuple[list[dict], LlmStats]:
    """spans(탐지 결과) → 검증된 수정 제안 목록. 호출 한도(QuotaExhausted)에 걸리면 지금까지 결과로 멈춘다(stats.stopped)."""
    from ..ocr import QuotaExhausted
    lim = lim or Limits()
    st = LlmStats()
    edits: list[dict] = []
    by_line: dict[tuple[int, int], list[dict]] = {}
    for s in spans:
        by_line.setdefault((s["page"], s["line"]), []).append(s)
    keys = sorted(by_line)
    span_by_id = {s["id"]: s for s in spans}
    accepted_ranges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for b in range(0, len(keys), lim.batch_lines):
        chunk = keys[b:b + lim.batch_lines]
        batch = [(k[0], by_line[k]) for k in chunk]
        items = build_items(batch, texts, cands)
        prompt = PROMPT.format(items=json.dumps(items, ensure_ascii=False, indent=1))
        ck = key_of("edit", prompt)
        raw_edits = None
        cached = cache.get(ck) if cache else None
        if cached is not None:
            st.cache_hits += 1
            try:
                raw_edits = _parse(cached)
            except (ValueError, json.JSONDecodeError):
                raw_edits = None
        if raw_edits is None:
            for attempt in range(1 + max_format_retries):
                try:
                    st.requests += 1
                    text = caller(prompt, schema=SCHEMA)
                except QuotaExhausted as e:
                    st.stopped = str(e)
                    log(f"[LLM 중단] {e}")
                    return edits, st
                try:
                    raw_edits = _parse(text)
                    if cache:
                        cache.put(ck, text)
                    break
                except (ValueError, json.JSONDecodeError) as e:
                    st.format_retries += 1
                    if attempt == max_format_retries:
                        st.errors.append(f"형식 오류로 {len(items)}줄 건너뜀: {e}")
                        log(f"  [LLM 형식 오류] 재시도 소진: {e}")
            if raw_edits is None:
                continue
        touched: set[str] = set()
        for r in raw_edits:
            sid = r.get("id") if isinstance(r, dict) else None
            span = span_by_id.get(sid) if isinstance(sid, str) else None
            if span is None:
                edits.append({"id": sid if isinstance(sid, str) else None, "page": None, "line": None, "source": "llm",
                              "status": "rejected", "code": "알수없는_id", "detail": "요청에 없던 항목 id",
                              "original": r.get("original") if isinstance(r, dict) else None,
                              "corrected": r.get("corrected") if isinstance(r, dict) else None})
                st.rejected += 1
                continue
            touched.add(sid)
            line = texts[span["page"]][span["line"] - 1]
            v = validate(span, line, r, lim, protect)
            if v["status"] in ("proposed", "held"):
                key = (span["page"], span["line"])
                rngs = accepted_ranges.setdefault(key, [])
                if len(rngs) >= lim.max_edits_per_line:
                    v = {**v, "status": "rejected", "code": "줄당_수정_초과", "detail": f"한 줄 {lim.max_edits_per_line}건 초과"}
                elif any(v["start"] < e2 and s2 < v["end"] for s2, e2 in rngs):
                    v = {**v, "status": "rejected", "code": "겹침", "detail": "같은 줄의 다른 수정과 겹침"}
                else:
                    rngs.append((v["start"], v["end"]))
            st.__dict__[{"proposed": "proposed", "held": "held", "rejected": "rejected", "nochange": "nochange"}[v["status"]]] += 1
            edits.append(v)
        st.unmatched_targets += sum(1 for s in (x for _, sp in batch for x in sp) if s["id"] not in touched)
    return edits, st
