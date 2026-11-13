"""M5 이미지 근거 검증 테스트 (가짜 Gemini: 이미지 내용 대신 미리 정한 '재판독' 문자열을 돌려준다)."""
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookpdf.ocr import QuotaExhausted  # noqa: E402
from bookpdf.proofread import verify as vf  # noqa: E402
from bookpdf.proofread.cache import ResponseCache  # noqa: E402


class FakeVision:
    def __init__(self, replies):
        self.replies, self.calls, self.prompts, self.images = list(replies), 0, [], []

    def __call__(self, prompt, *, image=None, mime="image/png", schema=None):
        self.calls += 1
        self.prompts.append(prompt)
        self.images.append(image)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r if isinstance(r, str) else json.dumps({"text": r}, ensure_ascii=False)


@pytest.fixture
def env(tmp_path):
    img = tmp_path / "p0001.png"
    Image.new("RGB", (800, 600), "white").save(img)
    line = "추세가 삼아 있다고 본다"
    page = {"page": 1, "image": str(img), "lines": [line], "blocks": None}
    span = {"id": "p0001-L001-C004", "page": 1, "line": 1, "col": 4, "end": 6, "token": "삼아", "box": [200, 100, 260, 130],
            "signals": [{"name": "low_conf", "weight": .5, "detail": ""}]}
    return tmp_path, page, span


def run(env, replies, cands=("살아",), span_extra=None, page_extra=None, cache=None):
    tmp, page, span = env
    sp = {**span, **(span_extra or {})}
    pg = {**page, **(page_extra or {})}
    f = FakeVision(replies)
    res, st = vf.verify_spans([sp], {sp["id"]: list(cands)}, {1: pg}, f, tmp / "crops", cache)
    return res[sp["id"]], st, f


def test_confirmed_when_reread_matches_candidate_not_original(env):
    r, st, f = run(env, ["추세가 살아 있다"])
    assert r["verdict"] == "confirmed" and r["matched"] == "살아" and r["mode"] == "word_box"
    assert st.confirmed == 1 and f.calls == 1
    assert (env[0] / "crops" / "p0001-L001-C004.png").exists() and r["crop"].endswith(".png")


def test_prompt_is_independent_no_ocr_text_or_candidates(env):
    r, st, f = run(env, ["살아"], cands=("살아", "삼각"))
    p = f.prompts[0]
    assert "삼아" not in p and "살아" not in p and "삼각" not in p        # 원문 OCR·후보를 알려주지 않음
    assert f.images[0][:8] == b"\x89PNG\r\n\x1a\n"


def test_supports_original_is_dismissed(env):
    r, st, _ = run(env, ["추세가 삼아 있다"])
    assert r["verdict"] == "supports_original" and st.supports_original == 1 and r["matched"] is None


@pytest.mark.parametrize("reply,verdict", [("추세가 샅아 있다", "mismatch"), ("완전히 다른 글자", "mismatch"),
                                           ("추세가 [판독불가] 있다", "unreadable"), ("", "unreadable"),
                                           ("삼아 살아", "mismatch")])           # 원문과 후보가 같이 보이면 모호 → 보류
def test_mismatch_unreadable_ambiguous(env, reply, verdict):
    r, st, _ = run(env, [reply])
    assert r["verdict"] == verdict and r["matched"] is None


def test_ambiguous_two_candidates(env):
    r, _, _ = run(env, ["살아 삶아"], cands=("살아", "삶아"))
    assert r["verdict"] == "mismatch"
    r, _, _ = run(env, ["거래량을"], cands=("거래", "거래량"))
    assert r["verdict"] == "confirmed" and r["matched"] == "거래량"           # 짧은 후보가 긴 후보에 포함되면 긴 쪽


def test_no_image_and_no_candidates_and_errors(env):
    tmp, page, span = env
    res, st = vf.verify_spans([span], {span["id"]: ["살아"]}, {1: {**page, "image": str(tmp / "없음.png")}}, FakeVision([]), tmp / "c")
    assert res[span["id"]]["verdict"] == "no_image" and st.no_image == 1
    res, st = vf.verify_spans([span], {span["id"]: []}, {1: page}, FakeVision([]), tmp / "c")
    assert res[span["id"]]["verdict"] == "mismatch" and "후보 없음" in res[span["id"]]["note"]
    r, st, _ = run(env, [RuntimeError("boom")])
    assert r["verdict"] == "error" and st.error == 1
    r, st, _ = run(env, ["{깨진 json"])
    assert r["verdict"] == "error"


def test_quota_stop_marks_rest_as_error(env):
    tmp, page, span = env
    span2 = {**span, "id": "p0001-L001-C010", "col": 10, "end": 12}
    f = FakeVision([QuotaExhausted("일일 한도(RPD) 도달", "daily")])
    res, st = vf.verify_spans([span, span2], {span["id"]: ["a"], span2["id"]: ["b"]}, {1: page}, f, tmp / "c")
    assert st.stopped and res[span["id"]]["verdict"] == "error" and res[span2["id"]]["verdict"] == "error" and st.error == 2


def test_line_est_from_blocks_and_page_anchor(env):
    tmp, page, span = env
    blocks = [{"text": "첫 줄\n추세가 삼아 있다고 본다", "x0": 50, "x1": 700, "y0": 100, "y1": 180}]
    pg = {**page, "lines": ["첫 줄", "추세가 삼아 있다고 본다"], "blocks": blocks, "markdown_original": "첫 줄\n추세가 삼아 있다고 본다"}
    sp = {**span, "box": None, "line": 2}
    f = FakeVision(["살아"])
    res, _ = vf.verify_spans([sp], {sp["id"]: ["살아"]}, {1: pg}, f, tmp / "c")
    r = res[sp["id"]]
    assert r["mode"] == "line_est" and r["verdict"] == "confirmed"
    box, mode = vf.locate_crop(pg, sp, (800, 600))
    assert mode == "line_est" and 120 <= box[1] < 140 and 180 < box[3] <= 200          # 두 번째 줄(140~180)에 여백을 더한 범위
    # blocks 없음 → 쪽 전체 + 앞뒤 낱말 안내
    sp2 = {**span, "box": None}
    f2 = FakeVision(["살아"])
    res, _ = vf.verify_spans([sp2], {sp2["id"]: ["살아"]}, {1: page}, f2, tmp / "c2")
    assert res[sp2["id"]]["mode"] == "page_anchor" and "추세가" in f2.prompts[0] and "삼아" not in f2.prompts[0] and "살아" not in f2.prompts[0]


def test_block_line_map_matches_markdown_layout():
    blocks = [{"text": "a\nb"}, {"text": "c"}]
    assert vf.block_line_map(blocks) == [(0, 0, 2), (0, 1, 2), None, (1, 0, 1)]
    assert len("\n\n".join(b["text"] for b in blocks).split("\n")) == 4


def test_cache_avoids_second_call(env, tmp_path):
    c = ResponseCache(tmp_path / "v.jsonl")
    run(env, ["살아"], cache=c)
    r, st, f = run(env, [], cache=ResponseCache(tmp_path / "v.jsonl"))
    assert f.calls == 0 and st.cache_hits == 1 and r["verdict"] == "confirmed"
