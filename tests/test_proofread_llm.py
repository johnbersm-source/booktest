"""M4 LLM 수정 목록 + Gemini 호출기 테스트. Gemini 는 전부 가짜 클라이언트."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_proofread_fixtures as fx  # noqa: E402
from bookpdf.ocr import QuotaExhausted  # noqa: E402
from bookpdf.proofread import candidates as cd, detect as dt, gemini, llm_edit as le, normalize as nz  # noqa: E402
from bookpdf.proofread.cache import ResponseCache  # noqa: E402

P = nz.Protect()


@pytest.fixture(scope="module")
def book():
    pages = [{"page": n, "text": nz.normalize_text("\n".join(l), P).text} for n, l in fx.noisy(fx.GOLD, fx.SUBS).items()]
    vocab = cd.build_vocab([p["text"] for p in pages], P)
    res = dt.detect_pages(pages, vocab, P)
    texts = {p["page"]: p["text"].split("\n") for p in pages}
    cands = {s["id"]: cd.gen_candidates(s, vocab) for s in res["spans"]}
    return pages, vocab, res["spans"], texts, cands


class Fake:
    """호출기 가짜: 요청마다 respond(prompt) 의 반환(dict/str)을 돌려준다."""

    def __init__(self, respond):
        self.respond, self.calls, self.prompts = respond, 0, []

    def __call__(self, prompt, *, image=None, mime="image/png", schema=None):
        self.calls += 1
        self.prompts.append(prompt)
        r = self.respond(prompt)
        return r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)


def span_of(spans, token):
    return next(s for s in spans if s["token"] == token)


def ids_in(prompt):
    import re
    return re.findall(r'"id": "(p\d{4}-L\d{3}-C\d{3})"', prompt)


def test_good_edits_are_proposed(book):
    _, _, spans, texts, cands = book
    truth = {"삼아": "살아", "R5I": "RSI", "거래랑을": "거래량을", "신뢰드가": "신뢰도가"}
    by_id = {s["id"]: truth[s["token"]] for s in spans if s["token"] in truth}

    def respond(prompt):
        return {"edits": [{"id": i, "original": next(s["token"] for s in spans if s["id"] == i), "corrected": by_id[i],
                           "reason": "OCR 오인식", "confidence": 0.9} for i in ids_in(prompt) if i in by_id]}
    f = Fake(respond)
    edits, st = le.propose_edits(spans, texts, cands, f, P)
    ok = {e["original"]: e["corrected"] for e in edits if e["status"] == "proposed"}
    assert ok == truth, (ok, [e for e in edits if e["status"] != "proposed"])
    assert st.proposed == 4 and st.rejected == 0
    # 위치가 줄 안에서 정확히 원문을 가리킨다
    for e in edits:
        assert texts[e["page"]][e["line"] - 1][e["start"]:e["end"]] == e["original"]


def test_prompt_contains_only_flagged_context(book):
    _, _, spans, texts, cands = book
    f = Fake(lambda p: {"edits": []})
    le.propose_edits(spans, texts, cands, f, P)
    p = f.prompts[0]
    assert "삼아" in p and "temperature" not in p and "맞춤법" in p       # 금지 규칙 명시
    assert "후보" in p or "candidates" in p
    assert all(len(x) < 20000 for x in f.prompts)


@pytest.mark.parametrize("original,corrected,code", [
    ("추세가 삼아 있다고 본다.", "추세가 살아 있음을 의미한다.", "길이변화"),           # 문장 재작성(과교정): 구간 밖/길이/거리
    ("삼아", "삼성전자가 오른다는 의미", "길이변화"),                                      # 없는 내용 지어내기
    ("존재하지않는원문", "다른말", "원문_불일치"),                                          # 줄에 없는 원문(환각)
    ("주가", "주식", "구간_밖"),                                                       # 다른 위치(의심 구간 밖) 수정
])
def test_overcorrection_and_hallucination_rejected(book, original, corrected, code):
    _, _, spans, texts, cands = book
    s = span_of(spans, "삼아")
    v = le.validate(s, texts[s["page"]][s["line"] - 1], {"original": original, "corrected": corrected, "confidence": 0.99},
                    le.Limits(), P)
    assert v["status"] == "rejected"
    if code == "길이변화":
        assert v["code"] in ("길이변화", "편집거리", "구간_밖")
    else:
        assert v["code"] == code


def test_edit_distance_and_ratio_limits(book):
    _, _, spans, texts, _ = book
    s = span_of(spans, "삼아")
    line = texts[s["page"]][s["line"] - 1]
    v = le.validate(s, line, {"original": "삼아", "corrected": "포도", "confidence": 0.99}, le.Limits(), P)
    assert v["status"] == "rejected" and v["code"] == "편집거리"
    assert le.validate(s, line, {"original": "삼아", "corrected": "살아", "confidence": 0.99}, le.Limits(), P)["status"] == "proposed"


def test_numeric_change_requires_numeric_signal():
    line = "지수는 3,000선을 넘었다"
    span = {"id": "p0001-L001-C004", "page": 1, "line": 1, "col": 4, "end": 10, "token": "3,000", "kind": "num",
            "signals": [{"name": "rare_near_freq", "weight": .4, "detail": ""}]}
    raw = {"original": "3,000", "corrected": "3,800", "confidence": 0.95}
    v = le.validate(span, line, raw, le.Limits(), P)
    assert v["status"] == "rejected" and v["code"] == "숫자영문_신호없음"
    span2 = {**span, "signals": [{"name": "ocr_disagree", "weight": .7, "detail": ""}]}
    assert le.validate(span2, line, raw, le.Limits(), P)["status"] == "proposed"
    # 영문도 마찬가지
    span3 = {**span, "token": "RSI", "col": 0, "end": 3, "signals": [{"name": "rare_near_freq", "weight": .4, "detail": ""}]}
    assert le.validate(span3, "RSI는", {"original": "RSI", "corrected": "RSX", "confidence": .9}, le.Limits(), P)["code"] == "숫자영문_신호없음"


def test_protected_term_not_editable():
    span = {"id": "p0001-L001-C000", "page": 1, "line": 1, "col": 0, "end": 4, "token": "MACD", "kind": "en",
            "signals": [{"name": "english_far", "weight": .5, "detail": ""}]}
    v = le.validate(span, "MACD는", {"original": "MACD", "corrected": "MACO", "confidence": .99}, le.Limits(), P)
    assert v["status"] == "rejected" and v["code"] == "보호어"


def test_spacing_only_edit_needs_signal():
    line = "그 는 이 곳 에서 본다"
    sig = {"name": "syllable_split", "weight": .5, "detail": ""}
    span = {"id": "p0001-L001-C000", "page": 1, "line": 1, "col": 0, "end": 11, "token": "그 는 이 곳 에", "kind": "split", "signals": [sig]}
    raw = {"original": "그 는 이 곳 에", "corrected": "그는 이곳에", "confidence": .9}
    assert le.validate(span, line, raw, le.Limits(), P)["status"] == "proposed"
    span2 = {**span, "signals": [{"name": "rare_near_freq", "weight": .5, "detail": ""}]}
    assert le.validate(span2, line, raw, le.Limits(), P)["code"] == "띄어쓰기_신호없음"


def test_low_confidence_is_held_not_rejected(book):
    _, _, spans, texts, _ = book
    s = span_of(spans, "삼아")
    v = le.validate(s, texts[s["page"]][s["line"] - 1], {"original": "삼아", "corrected": "살아", "confidence": 0.4}, le.Limits(), P)
    assert v["status"] == "held" and v["code"] == "낮은확신"


def test_format_violations_rejected(book):
    _, _, spans, texts, _ = book
    s = span_of(spans, "삼아")
    line = texts[s["page"]][s["line"] - 1]
    for raw in ({"original": 1, "corrected": "x", "confidence": .9}, {"original": "삼아", "corrected": "살아", "confidence": "높음"},
                {"original": "삼아", "corrected": "살아\n추가", "confidence": .9}, {"original": "삼아", "corrected": "살아", "confidence": 3}):
        v = le.validate(s, line, raw, le.Limits(), P)
        assert v["status"] == "rejected" and v["code"] == "형식", raw


def test_unknown_id_overlap_and_line_cap(book):
    _, _, spans, texts, cands = book
    s = span_of(spans, "삼아")

    def respond(prompt):
        return {"edits": [
            {"id": "p9999-L001-C000", "original": "가", "corrected": "나", "reason": "", "confidence": .9},
            {"id": s["id"], "original": "삼아", "corrected": "살아", "reason": "", "confidence": .9},
            {"id": s["id"], "original": "삼아", "corrected": "산아", "reason": "", "confidence": .9},      # 같은 위치 겹침
        ]}
    edits, st = le.propose_edits([s], texts, cands, Fake(respond), P)
    codes = [e.get("code") for e in edits]
    assert codes == ["알수없는_id", "", "겹침"] and st.proposed == 1 and st.rejected == 2


def test_bad_json_retried_then_recovered_and_gives_up(book):
    _, _, spans, texts, cands = book
    s = span_of(spans, "삼아")
    seq = iter(["이건 JSON 이 아님", '{"edits": 3}', json.dumps({"edits": [{"id": s["id"], "original": "삼아", "corrected": "살아", "confidence": .9}]})])
    f = Fake(lambda p: next(seq))
    edits, st = le.propose_edits([s], texts, cands, f, P, max_format_retries=2)
    assert st.format_retries == 2 and f.calls == 3 and st.proposed == 1
    f2 = Fake(lambda p: "깨짐")
    edits, st = le.propose_edits([s], texts, cands, f2, P, max_format_retries=1)
    assert edits == [] and st.errors and f2.calls == 2          # 죽지 않고 건너뜀


def test_quota_stop_returns_partial(book):
    _, _, spans, texts, cands = book
    state = {"n": 0}

    def respond(prompt):
        state["n"] += 1
        if state["n"] > 1:
            raise QuotaExhausted("일일 한도(RPD) 도달", "daily")
        return {"edits": []}
    lim = le.Limits(batch_lines=1)
    edits, st = le.propose_edits(spans, texts, cands, Fake(respond), P, lim)
    assert st.stopped and "일일" in st.stopped and st.requests == 2


def test_cache_prevents_repeat_calls(book, tmp_path):
    _, _, spans, texts, cands = book
    f = Fake(lambda p: {"edits": []})
    c = ResponseCache(tmp_path / "c.jsonl")
    le.propose_edits(spans, texts, cands, f, P, cache=c)
    n = f.calls
    f2 = Fake(lambda p: {"edits": []})
    _, st = le.propose_edits(spans, texts, cands, f2, P, cache=ResponseCache(tmp_path / "c.jsonl"))
    assert f2.calls == 0 and st.cache_hits == n


# ---------------- Gemini 호출기(가짜 클라이언트로 재시도·한도 로직) ----------------
class _Resp:
    text = '{"edits": []}'


class _Err(Exception):
    def __init__(self, msg, code):
        super().__init__(msg)
        self.code = code


class _Models:
    def __init__(self, behaviors):
        self.behaviors, self.seen = list(behaviors), []

    def generate_content(self, model, contents, config):
        self.seen.append((model, config))
        b = self.behaviors.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


class _Client:
    def __init__(self, behaviors):
        self.models = _Models(behaviors)


def test_gemini_caller_temperature_zero_and_retry_429():
    sleeps = []
    cl = _Client([_Err("429 RESOURCE_EXHAUSTED PerMinute retry in 2s", 429), _Resp()])
    g = gemini.GeminiCaller("k", "m", rpm=0, client=cl, sleep=sleeps.append, log=lambda *_: None)
    assert g("프롬프트") == '{"edits": []}'
    assert g.calls == 2 and sleeps and sleeps[0] >= 3
    model, cfg = cl.models.seen[-1]
    assert model == "m" and cfg.temperature == 0


def test_gemini_caller_daily_quota_stops_and_max_calls():
    cl = _Client([_Err("429 RESOURCE_EXHAUSTED RequestsPerDay", 429)])
    g = gemini.GeminiCaller("k", "m", rpm=0, client=cl, sleep=lambda s: None, log=lambda *_: None)
    with pytest.raises(QuotaExhausted) as ei:
        g("x")
    assert ei.value.kind == "daily" and g.calls == 1
    g2 = gemini.GeminiCaller("k", "m", rpm=0, client=_Client([_Resp(), _Resp()]), max_calls=1, sleep=lambda s: None)
    g2("a")
    with pytest.raises(QuotaExhausted):
        g2("b")


def test_gemini_default_model_and_env(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "  abc ")
    s = gemini.load_settings()
    assert s.model == "gemini-3.5-flash-lite" and s.api_key == "abc"
    monkeypatch.setenv("GEMINI_MODEL", "other-model")
    assert gemini.load_settings().model == "other-model"
