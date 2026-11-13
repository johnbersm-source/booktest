"""M6 파이프라인(판정·적용·되돌리기·검수·리포트) 통합 테스트. Gemini 는 전부 가짜, 데이터는 합성."""
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_proofread_fixtures as fx  # noqa: E402
from bookpdf.proofread import apply as ap, normalize as nz, pipeline as pl, report as rp, util  # noqa: E402

TRUTH = fx.TRUTH
NOISY = fx.noisy(fx.GOLD, fx.SUBS)


class Scripted:
    """텍스트 요청(이미지 없음)과 이미지 요청을 구분해 답하는 가짜 Gemini.

    text_mode: 'truth'(정답 제안) | 'overcorrect'(과교정·환각 시도) | 'wrong'(그럴듯하지만 틀린 제안) | 'none'
    vision_mode: 'truth'(정답 재판독) | 'original'(원문 그대로) | 'wrong' | 'unreadable'
    이미지 요청은 쪽 앵커(앞뒤 낱말)로 어느 구간인지 알아낸다(쪽 이미지는 합성이라 실제로 읽을 수 없음).
    """

    def __init__(self, text_mode="truth", vision_mode="truth"):
        self.text_mode, self.vision_mode, self.calls = text_mode, vision_mode, 0
        self.text_prompts, self.vision_prompts = [], []

    def __call__(self, prompt, *, image=None, mime="image/png", schema=None):
        self.calls += 1
        if image is None:
            self.text_prompts.append(prompt)
            items = json.loads(prompt[prompt.index("검토 항목(JSON):") + len("검토 항목(JSON):"):])
            edits = []
            for it in items:
                for t in it["targets"]:
                    tok = t["target"]
                    if self.text_mode == "truth" and tok in TRUTH:
                        edits.append({"id": t["id"], "original": tok, "corrected": TRUTH[tok], "reason": "OCR 오인식", "confidence": 0.92})
                    elif self.text_mode == "wrong" and tok in TRUTH:
                        edits.append({"id": t["id"], "original": tok, "corrected": tok[:-1] + "별", "reason": "?", "confidence": 0.9})
                    elif self.text_mode == "overcorrect":
                        edits.append({"id": t["id"], "original": it["line"], "corrected": it["line"] + " 그리고 새로운 문장을 지어냅니다.",
                                      "reason": "문장 다듬기", "confidence": 0.99})
            return json.dumps({"edits": edits}, ensure_ascii=False)
        self.vision_prompts.append(prompt)
        m = re.search(r'왼쪽 글자 "(.*?)" 바로 다음부터 오른쪽 글자 "(.*?)" 바로 앞까지', prompt)
        assert m, "이 테스트의 합성 쪽에는 tsv/blocks 가 없으므로 page_anchor 모드여야 함"
        left, right = m.group(1), m.group(2)
        for n, lines in NOISY.items():
            for ln in lines:
                key = (left + " ").lstrip() if left != "(줄 처음)" else ""
                for bad, good in TRUTH.items():
                    if bad in ln and (left == "(줄 처음)" or ln.split(bad)[0].rstrip().endswith(left.split()[-1] if left.split() else "")):
                        return json.dumps({"text": {"truth": good, "original": bad, "wrong": bad[:-1] + "별", "unreadable": "[판독불가]"}[self.vision_mode]},
                                          ensure_ascii=False)
        return json.dumps({"text": "알 수 없음"})


def sha_dir(d: Path) -> dict:
    return {p.name: hashlib.sha1(p.read_bytes()).hexdigest() for p in sorted(d.iterdir()) if p.is_file()}


@pytest.fixture
def work(tmp_path):
    return fx.make_work(tmp_path, NOISY)


def run(work, caller=None, **kw):
    opts = pl.Options(work=work, **kw)
    return opts, pl.run(opts, caller, log=lambda *a: None)


def by_token(res):
    return {d["token"]: d for d in res["decisions"]}


def test_full_run_adopts_only_with_image_evidence_and_keeps_originals(work):
    before = sha_dir(work / "pages")
    f = Scripted()
    opts, res = run(work, f)
    d = by_token(res)
    for bad, good in TRUTH.items():
        if bad in d:
            assert d[bad]["status"] == "adopted" and d[bad]["edit"]["after"] == good
            assert "이미지 재판독 일치" in d[bad]["evidence"]
    assert res["stats"]["adopted"] == 4 and res["stats"]["held"] == 0
    assert res["stats"]["api_calls"]["verify_requests"] == 4
    out = pl.apply_run(opts, res["state"], res["decisions"])
    # 원본 보존, 교정본은 corrected/ 에
    assert sha_dir(work / "pages") == before
    c1 = (work / "corrected" / "p0001.md").read_text(encoding="utf-8")
    assert "추세가 살아 있다고" in c1 and "삼아" not in c1
    assert "RSI는" in (work / "corrected" / "p0002.md").read_text(encoding="utf-8")
    assert "거래량을 함께" in (work / "corrected" / "p0003.md").read_text(encoding="utf-8")
    assert "2022년에는" in (work / "corrected" / "p0003.md").read_text(encoding="utf-8")        # M1 정규화로 고친 숫자
    j = json.loads((work / "corrected" / "p0001.json").read_text(encoding="utf-8"))
    assert j["markdown"] == c1 and j["proofread"]["edits"] == 1 and j["image"]
    assert out["edits"] == 4


def test_correction_log_reverts_exactly(work):
    opts, res = run(work, Scripted())
    pl.apply_run(opts, res["state"], res["decisions"])
    ok = ap.revert_dir(work / "corrected", work / "restored")
    assert all(ok.values()) and len(ok) == 5
    for n, lines in NOISY.items():
        orig = (work / "pages" / f"p{n:04d}.md").read_text(encoding="utf-8")
        assert (work / "restored" / f"p{n:04d}.md").read_text(encoding="utf-8") == orig
    # 교정본을 손으로 고치면 되돌리기는 안전하게 거부한다
    p = work / "corrected" / "p0001.md"
    p.write_text(p.read_text(encoding="utf-8") + "수동편집", encoding="utf-8")
    ok = ap.revert_dir(work / "corrected", work / "restored2")
    assert ok[1] is False and (work / "restored2" / "revert_errors.txt").exists()


def test_overcorrection_attempts_are_rejected_and_not_applied(work):
    opts, res = run(work, Scripted(text_mode="overcorrect", vision_mode="truth"))
    st = res["stats"]
    assert st["adopted"] == 4                     # 이미지+후보 근거로는 채택, LLM 의 과교정 제안 자체는 전부 거절
    assert st["rejected_proposals"]["total"] >= 4
    assert set(st["rejected_proposals"]["by_code"]) <= {"길이변화", "편집거리", "구간_밖", "원문_불일치"}
    pl.apply_run(opts, res["state"], res["decisions"])
    assert "새로운 문장" not in "".join((work / "corrected" / f"p{n:04d}.md").read_text(encoding="utf-8") for n in range(1, 6))


def test_llm_vs_image_conflict_goes_to_hold(work):
    opts, res = run(work, Scripted(text_mode="wrong", vision_mode="truth"))
    d = by_token(res)
    for bad in TRUTH:
        if bad in d:
            assert d[bad]["status"] == "held" and "충돌" in d[bad]["why"]
    assert res["stats"]["adopted"] == 0


def test_image_supports_original_is_dismissed(work):
    opts, res = run(work, Scripted(text_mode="truth", vision_mode="original"))
    assert res["stats"]["dismissed"] == 4 and res["stats"]["adopted"] == 0 and res["stats"]["held"] == 0
    pl.apply_run(opts, res["state"], res["decisions"])
    assert "삼아" in (work / "corrected" / "p0001.md").read_text(encoding="utf-8")        # 고치지 않음


def test_unreadable_or_wrong_image_reading_holds(work):
    for mode in ("unreadable", "wrong"):
        _, res = run(work, Scripted(vision_mode=mode))
        assert res["stats"]["adopted"] == 0 and res["stats"]["held"] == 4
        h = res["decisions"][0]
        assert h["hid"].startswith("H") and h["options"] and "이미지" in h["why"]


def test_no_verify_requires_two_independent_sources(work):
    # LLM 정답 + 책 어휘 후보 일치 → 채택 (숫자/영문 아님)
    opts, res = run(work, Scripted(), no_verify=True)
    d = by_token(res)
    assert d["삼아"]["status"] == "adopted" and "책 어휘 후보 일치" in d["삼아"]["evidence"]
    assert d["거래랑을"]["status"] == "adopted" and d["신뢰드가"]["status"] == "adopted"
    assert res["stats"]["api_calls"]["verify_requests"] == 0
    # 영문(R5I→RSI)은 숫자/영문이므로 두 번째 OCR 일치 없이는 채택하지 않음... 단 numeric 플래그는 숫자·영문 변경에만 붙는다
    assert d["R5I"]["status"] in ("adopted", "held")
    if d["R5I"]["status"] == "adopted":
        assert "책 어휘 후보 일치" in d["R5I"]["evidence"]


def test_no_verify_numeric_needs_alt_ocr(tmp_path):
    pages = {1: ["보고서의 매출은 1982년 기록이다. 매출 보고서 보고서 매출 보고서"]}
    work = fx.make_work(tmp_path / "a", pages)
    alt = fx.make_work(tmp_path / "b", {1: ["보고서의 매출은 1992년 기록이다. 매출 보고서 보고서 매출 보고서"]})

    class OnlyNumeric:
        calls = 0

        def __call__(self, prompt, *, image=None, mime="image/png", schema=None):
            items = json.loads(prompt[prompt.index("검토 항목(JSON):") + len("검토 항목(JSON):"):])
            edits = [{"id": t["id"], "original": t["target"], "corrected": "1992", "reason": "", "confidence": .95}
                     for it in items for t in it["targets"] if t["target"] == "1982"]
            return json.dumps({"edits": edits})
    # 두 번째 OCR 없이: 숫자 변경은 LLM 단독이므로 채택 불가(탐지 자체가 안 될 수도 있음)
    _, res0 = run(work, OnlyNumeric(), no_verify=True)
    assert res0["stats"]["adopted"] == 0
    # 두 번째 OCR 이 1992 로 읽음 → 탐지(ocr_disagree) + LLM 일치 → 채택
    opts, res = run(work, OnlyNumeric(), no_verify=True, alt_work=alt)
    d = by_token(res)
    assert d["1982"]["status"] == "adopted" and "두 번째 OCR 일치" in d["1982"]["evidence"]
    assert res["stats"]["mode"]["alt_pages"] == 1


def test_no_llm_no_verify_adopts_nothing_weak_and_writes_review(work):
    opts, res = run(work, None, no_llm=True, no_verify=True)
    assert res["stats"]["adopted"] == 0 and res["stats"]["held"] == 4 and res["stats"]["api_calls"]["total"] == 0
    out = pl.apply_run(opts, res["state"], res["decisions"])
    assert out["edits"] == 0
    # 정규화만 적용됨(2O22→2022), 나머지 글자는 원문 그대로
    assert "삼아" in (work / "corrected" / "p0001.md").read_text(encoding="utf-8")
    assert "2022년에는" in (work / "corrected" / "p0003.md").read_text(encoding="utf-8")


def test_review_file_choose_keep_custom_and_errors(work):
    opts, res = run(work, None, no_llm=True, no_verify=True)
    held = ap.held_view(res["decisions"])
    ap.write_review_template(work / "rev.txt", held)
    text = (work / "rev.txt").read_text(encoding="utf-8")
    assert "H001 =" in text and "t:" in text.splitlines()[1] + text
    hid = {h["original"]: h["hid"] for h in held}
    answers = (f"{hid['삼아']} = 1\n{hid['R5I']} = x\n{hid['거래랑을']} = t:거래량을\n{hid['신뢰드가']} = 9\n"
               f"H999 = 1\n쓰레기 줄\n")
    parsed, errs = ap.parse_review(answers, held)
    assert parsed[hid["삼아"]]["text"] == "살아" and parsed[hid["R5I"]]["action"] == "keep"
    assert parsed[hid["거래랑을"]] == {"action": "custom", "text": "거래량을"}
    assert hid["신뢰드가"] not in parsed and len(errs) == 3
    pl.apply_run(opts, res["state"], res["decisions"], parsed)
    c = lambda n: (work / "corrected" / f"p{n:04d}.md").read_text(encoding="utf-8")
    assert "살아 있다고" in c(1) and "R5I" in c(2) and "거래량을 함께" in c(3) and "신뢰드가" in c(5)
    log = json.loads((work / "corrected" / "corrections_log.json").read_text(encoding="utf-8"))
    srcs = {e["source"] for p in log["pages"].values() for e in p["edits"]}
    assert srcs == {"review:choose", "review:custom"}
    assert all(ap.revert_dir(work / "corrected", work / "r").values())


def test_multiple_edits_same_line_and_revert():
    text = "가나다 라마바 사아자"
    edits = [{"line": 1, "start": 0, "end": 3, "before": "가나다", "after": "가나"},
             {"line": 1, "start": 8, "end": 11, "before": "사아자", "after": "사아자차카"},
             {"line": 1, "start": 4, "end": 7, "before": "라마바", "after": "라"}]
    new = ap.apply_edits_to_text(text, edits)
    assert new == "가나 라 사아자차카"
    assert ap.revert_edits_in_text(new, edits) == text
    with pytest.raises(ValueError):
        ap.apply_edits_to_text(text, [{"line": 1, "start": 0, "end": 2, "before": "XX", "after": "a"}])
    with pytest.raises(ValueError):
        ap.apply_edits_to_text(text, [{"line": 1, "start": 0, "end": 3, "before": "가나다", "after": "a"},
                                      {"line": 1, "start": 2, "end": 5, "before": "다 라", "after": "b"}])


def test_report_html_and_md_mobile_friendly_and_escaped(work):
    opts, res = run(work, Scripted(vision_mode="unreadable"))
    pl.apply_run(opts, res["state"], res["decisions"])
    # 악성 문자열이 문맥에 있어도 이스케이프되는지
    res["decisions"][0]["context"] = res["decisions"][0]["context"] + " <script>alert(1)</script>"
    paths = rp.write_reports(res["state"]["out"], res["stats"], res["decisions"], res["detections"]["page_notes"], res["llm"])
    h = paths["html"].read_text(encoding="utf-8")
    assert "viewport" in h and "<script>alert" not in h and "&lt;script&gt;" in h and "<table" not in h
    assert "prefers-color-scheme" in h and "보류 항목 4건" in h and "crops/" in h
    m = paths["md"].read_text(encoding="utf-8")
    assert "## 요약" in m and "H001" in m and "|---|" not in m
    assert (res["state"]["out"] / "crops").is_dir() and any((res["state"]["out"] / "crops").iterdir())
    assert paths["review"].read_text(encoding="utf-8").count(" =\n") == 4


def test_resume_uses_cache_and_no_second_api_calls(work):
    f1 = Scripted()
    run(work, f1)
    n = f1.calls
    f2 = Scripted()
    _, res = run(work, f2)
    assert f2.calls == 0 and res["stats"]["adopted"] == 4 and res["stats"]["api_calls"]["cache_hits"] >= n


def test_pages_subset_and_missing_page_error(work):
    _, res = run(work, Scripted(), pages=[2, 3])
    assert res["stats"]["pages"] == 2 and {d["page"] for d in res["decisions"]} <= {2, 3}
    with pytest.raises(ValueError):
        run(work, Scripted(), pages=[2, 77])


def test_progress_and_outputs_exist(work):
    opts, res = run(work, Scripted())
    out = res["state"]["out"]
    for name in ("detections.json", "candidates.json", "llm_edits.json", "verify.json", "decisions.json", "stats.json", "progress.json"):
        assert (out / name).exists(), name
    assert set(json.loads((out / "progress.json").read_text())["stages"]) >= {"normalize", "detect", "llm", "verify", "decide"}
    assert (out / "glossary_auto.txt").exists()
