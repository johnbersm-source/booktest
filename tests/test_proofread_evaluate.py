"""M7 평가 테스트 (합성 정답 쪽)."""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_proofread_fixtures as fx  # noqa: E402
import test_proofread_pipeline as tp  # noqa: E402
from bookpdf.proofread import evaluate as ev, pipeline as pl, util  # noqa: E402


def test_myers_matches_naive_levenshtein():
    rnd = random.Random(1)
    for _ in range(200):
        a = "".join(rnd.choice("가나다라마") for _ in range(rnd.randint(0, 30)))
        b = "".join(rnd.choice("가나다라마") for _ in range(rnd.randint(0, 30)))
        assert util.myers_distance(a, b) == util.levenshtein(a, b), (a, b)
    assert util.myers_distance(list("ab cd".split()), ["ab", "x"]) == 1
    long_a, long_b = "가나다" * 3000, "가나라" * 3000
    assert util.myers_distance(long_a, long_b) == 3000        # 긴 글도 즉시 계산


def test_cer_wer_basics():
    assert ev.cer("가나다라", "가나다라") == (0, 4)
    assert ev.cer("가나다라", "가나라") == (2, 4) or ev.cer("가나다라", "가나라")[0] == 1
    assert ev.cer("가 나\n다", "가 나 다")[0] == 0              # 줄바꿈/공백 정리
    assert ev.cer("가 나 다", "가나다", ignore_space=True)[0] == 0
    assert ev.wer("추세가 살아 있다", "추세가 삼아 있다") == (1, 3)
    assert ev.cer("", "x") == (1, 0) and ev._rate(1, 0) is None


@pytest.fixture
def evaluated(tmp_path):
    work = fx.make_work(tmp_path, tp.NOISY)
    gold = tmp_path / "gold"
    gold.mkdir()
    for n, lines in fx.GOLD.items():
        if n != 4:                                  # 4쪽 정답은 일부러 없음
            (gold / f"p{n:04d}.md").write_text("\n".join(lines), encoding="utf-8")
    opts = pl.Options(work=work, use_tsv=False)
    res = pl.run(opts, tp.Scripted(), log=lambda *a: None)
    pl.apply_run(opts, res["state"], res["decisions"])
    return work, gold, res


def test_cer_wer_before_after_improves(evaluated):
    work, gold, res = evaluated
    r = ev.evaluate(work, work / "corrected", gold)
    s = r["summary"]
    assert s["compared_pages"] == 4 and s["gold_pages"] == 4 and s["caveat"]              # 정답 4쪽 → 참고용 경고
    assert s["cer_after"] < s["cer_before"] and s["wer_after"] < s["wer_before"]
    assert s["cer_after"] <= s["cer_normalized_only"] <= s["cer_before"]                   # 정규화 → 수정 순으로 개선
    assert s["cer_after"] == 0.0                                                            # 합성 데이터: 모든 오류가 정확히 고쳐짐
    assert s["pages_worse"] == 0 and s["pages_improved"] >= 3
    assert {p["page"] for p in r["pages"]} == {1, 2, 3, 5}


def test_gold_check_flags_wrong_edits(evaluated):
    work, gold, res = evaluated
    chk = ev.evaluate(work, work / "corrected", gold)["gold_check"]
    assert chk["wrong"] == 0 and chk["right"] >= 3 and chk["wrong_rate"] == 0.0
    # 틀리게 고치는 가짜 LLM+가짜 이미지(둘 다 같은 오답)를 만들면 gold_check 가 잡아야 한다
    class Bad(tp.Scripted):
        def __call__(self, prompt, *, image=None, **kw):
            r = super().__call__(prompt, image=image, **kw)
            return r.replace("살아", "산아").replace("신뢰도가", "신뢰도카")
    opts = pl.Options(work=work, corrected=work / "corrected_bad", out=work / "proofread_bad", use_tsv=False)
    res2 = pl.run(opts, Bad(), log=lambda *a: None)
    pl.apply_run(opts, res2["state"], res2["decisions"])
    chk2 = ev.evaluate(work, work / "corrected_bad", gold)["gold_check"]
    assert chk2["wrong"] >= 1 and chk2["wrong_rate"] > 0
    assert ev.evaluate(work, work / "corrected_bad", gold)["summary"]["cer_after"] > 0


def test_audit_sample_is_random_reproducible_and_scored(evaluated):
    work, gold, res = evaluated
    items = ev.sample_audit(work / "corrected", work / "proofread" / "decisions.json", 3, seed=7)
    assert len(items) == 3 and {i["aid"] for i in items} == {"A001", "A002", "A003"}
    assert items == ev.sample_audit(work / "corrected", work / "proofread" / "decisions.json", 3, seed=7)
    assert all(i["after"] != i["before"] and i["context"] for i in items)
    paths = ev.write_audit(work / "audit", items)
    assert "A001 =" in paths["answers"].read_text(encoding="utf-8")
    sc = ev.score_audit("A001 = o\nA002 = x\nA003 = ?\n쓰레기\nA999 = o\n", items)
    assert sc["right"] == 1 and sc["wrong"] == 1 and sc["undecided"] == 1 and sc["wrong_rate"] == 0.5 and len(sc["errors"]) == 2
    lo, hi = sc["wilson95"]
    assert 0 <= lo < 0.5 < hi <= 1                          # 표본이 작아 구간이 넓다
    assert ev.score_audit("", items)["wrong_rate"] is None


def test_audit_request_larger_than_pool(evaluated):
    work, *_ = evaluated
    assert len(ev.sample_audit(work / "corrected", None, 99, seed=1)) == 4


def test_missing_gold_pages_reported(evaluated, tmp_path):
    work, gold, _ = evaluated
    (gold / "p0099.md").write_text("없는 쪽", encoding="utf-8")
    s = ev.evaluate(work, work / "corrected", gold)["summary"]
    assert s["missing_pages"] == [99] and s["compared_pages"] == 4
