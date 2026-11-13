"""M2 탐지 + M3 후보 테스트 (합성 데이터)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_proofread_fixtures as fx  # noqa: E402
from bookpdf.proofread import candidates as cd, detect as dt, normalize as nz, util  # noqa: E402

P = nz.Protect()


def prep(pages):
    """쪽 {n: [줄]} → (정규화된 pages, vocab). 실제 사용과 같이 어휘는 (오류 포함) 책 자체에서 만든다."""
    out = []
    for n, lines in pages.items():
        out.append({"page": n, "text": nz.normalize_text("\n".join(lines), P).text})
    vocab = cd.build_vocab([p["text"] for p in out], P)
    return out, vocab


def tokens(res):
    return {s["token"]: s for s in res["spans"]}


@pytest.fixture(scope="module")
def noisy_book():
    pages, vocab = prep(fx.noisy(fx.GOLD, fx.SUBS))
    return pages, vocab, dt.detect_pages(pages, vocab, P)


def test_normalize_already_fixed_digit_noise(noisy_book):
    pages, _, _ = noisy_book
    assert "2022년에는" in pages[2]["text"] and "2O22" not in pages[2]["text"]     # M1 이 숫자 문맥 혼동을 이미 고침


def test_detects_injected_errors(noisy_book):
    _, _, res = noisy_book
    t = tokens(res)
    for bad in ("삼아", "R5I", "거래랑을", "신뢰드가"):
        assert bad in t, f"{bad} 미탐지: {sorted(t)}"
    assert "rare_near_freq" in {s["name"] for s in t["삼아"]["signals"]}
    assert "english_far" in {s["name"] for s in t["R5I"]["signals"]}
    for s in res["spans"]:
        assert 0 < s["score"] <= 1 and s["id"].startswith("p") and s["context"]


def test_clean_book_has_few_false_positives():
    pages, vocab = prep(fx.GOLD)
    res = dt.detect_pages(pages, vocab, P)
    flagged = {s["token"] for s in res["spans"]}
    # 정답 책에서는 주입 오류 토큰이 나오지 않아야 하고, 전체 의심 구간도 적어야 한다
    assert not flagged & set(fx.TRUTH)
    assert len(res["spans"]) <= 3, flagged


def test_candidates_contain_truth(noisy_book):
    _, vocab, res = noisy_book
    t = tokens(res)
    for bad in ("삼아", "R5I", "거래랑을", "신뢰드가"):
        cands = cd.gen_candidates(t[bad], vocab)
        texts = [c["text"] for c in cands]
        assert fx.TRUTH[bad] in texts, (bad, cands)
        top = cands[0]
        assert {"freq", "dist", "wdist", "source", "score"} <= set(top)      # 근거 기록


def test_candidate_josa_reattached():
    v = cd.build_vocab(["거래량 거래량 거래량"], P)
    c = v.lookup_ko("거래랑을")
    assert c and c[0]["text"] == "거래량을"


def test_confusable_jamo_ranks_closer():
    assert util.weighted_jamo_distance("거래량", "거래랑") < util.weighted_jamo_distance("거래량", "거래강")
    assert util.jamo_distance("거래량", "거래랑") == 1


def test_no_candidate_invented_when_vocab_lacks_it():
    v = cd.build_vocab(["이동평균선 이동평균선 이동평균선"], P)
    assert v.lookup_ko("전혀다른말") == []


def test_year_range():
    sp = dt._Spans()
    for i, ln in enumerate(("1202년 사건", "20221년", "2023년 괜찮다", "3년 걸렸다", "2200년"), 1):
        dt.sig_year(sp, 1, i, ln, dt.Params())
    got = {s["token"] for s in sp.result(0.3)}
    assert got == {"1202", "20221", "2200"}


def test_caption_sequence_outlier_with_suggestion():
    lines = [(1, 1, "그림 1-1 가"), (2, 1, "그림 1-2 나"), (3, 1, "그림 1-8 다"), (4, 1, "그림 1-4 라"), (5, 1, "표 1 마"), (5, 2, "표 2 바")]
    sp = dt._Spans()
    dt.sig_caption(sp, lines, dt.Params())
    res = sp.result(0.3)
    assert [s["token"] for s in res] == ["1-8"]
    assert res[0]["suggest"] == [{"text": "1-3", "source": "sequence"}]
    # 번호 체계가 통째로 바뀐 경우(1,2,3,10,11,12)는 의심하지 않음
    sp = dt._Spans()
    dt.sig_caption(sp, [(i, 1, f"그림 {n} x") for i, n in enumerate([1, 2, 3, 10, 11, 12])], dt.Params())
    assert sp.result(0.3) == []


def test_table_sum_mismatch():
    ok = ["| 구분 | 비중 |", "|---|---|", "| 주식 | 40 |", "| 채권 | 35 |", "| 현금 | 25 |", "| 합계 | 100 |"]
    sp = dt._Spans()
    dt.sig_table_sums(sp, 1, ok)
    assert sp.result(0.1) == []
    bad = ok[:3] + ["| 채권 | 38 |"] + ok[4:]
    sp = dt._Spans()
    dt.sig_table_sums(sp, 1, bad)
    toks = {(s["line"], s["token"]) for s in sp.result(0.1)}
    assert (6, "100") in toks and (4, "38") in toks
    assert all("합 103" in s["signals"][0]["detail"] or "103" in s["signals"][0]["detail"] for s in sp.result(0.1))


def test_char_signals():
    sp = dt._Spans()
    for ln in ("한l글 은 좋다", "그 는 이 곳 에서", "읽을 수 없음 [판독불가] 끝", "깨진 � 글자 ㅁ 홀로"):
        dt.sig_chars(sp, 1, 1, ln)
    names = {s["signals"][0]["name"] for s in sp.result(0.3)}
    assert names == {"mixed_script", "syllable_split", "unreadable", "bad_char"}
    assert not any(s["token"] == "제1장" for s in sp.result(0.3))      # '제1장' 같은 서수는 의심하지 않음
    split = next(s for s in sp.result(0.3) if s["kind"] == "split")
    assert split["suggest"][0]["text"].startswith("그는이곳에") or split["suggest"][0]["source"] == "join"


def test_tsv_low_conf_and_box(tmp_path):
    pages = [{"page": 1, "text": "추세가 삼아 있다.\n거래량을 본다"}]
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
    rows = [(1, 1, 1, 1, 1, 10, 20, 50, 20, 95, "추세가"), (1, 1, 1, 1, 2, 70, 20, 40, 20, 31, "삼아"),
            (1, 1, 1, 1, 3, 120, 20, 40, 20, 90, "있다."), (1, 1, 2, 1, 1, 10, 60, 60, 20, 92, "거래량을"),
            (1, 1, 2, 1, 2, 80, 60, 40, 20, 90, "본다")]
    tsv = header + "\n".join("5\t" + "\t".join(map(str, r[:5])) + "\t" + "\t".join(map(str, r[5:10])) + f"\t{r[9]}\t{r[10]}"
                             if False else
                             f"5\t1\t{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[5]}\t{r[6]}\t{r[7]}\t{r[8]}\t{r[9]}\t{r[10]}" for r in rows)
    words = dt.parse_word_tsv(tsv)
    assert [w["conf"] for w in words] == [95, 31, 90, 92, 90]
    v = cd.build_vocab(["추세가 살아 있다 추세가 살아"], P)
    res = dt.detect_pages(pages, v, P, tsv={1: words})
    s = tokens(res)["삼아"]
    names = {x["name"] for x in s["signals"]}
    assert {"low_conf", "rare_near_freq"} <= names
    assert s["box"] == [70, 20, 110, 40] and s["line_box"][0] == 10 and s["score"] > 0.6
    assert res["stats"]["tsv_pages"] == 1


def test_load_tsv_uses_cache(tmp_path):
    cache = tmp_path / "tsv"
    cache.mkdir()
    (cache / "p0001.tsv").write_text("level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
                                     "5\t1\t1\t1\t1\t1\t1\t2\t3\t4\t88\t글\n", encoding="utf-8")
    w = dt.load_tsv({"page": 1, "image": None}, cache, run=False)
    assert w == [{"text": "글", "conf": 88.0, "box": [1, 2, 4, 6], "key": (1, 1, 1)}]
    assert dt.load_tsv({"page": 2, "image": None}, cache, run=False) is None


def test_two_ocr_disagreement():
    a = [{"page": 1, "text": "추세가 삼아 있다 2022년 매출"}]
    alt = {1: "추세가 살아 있다 2023년 매출"}
    v = cd.build_vocab(["추세가 살아 있다 추세가 살아 있다"], P)
    res = dt.detect_pages(a, v, P, alt_texts=alt)
    t = tokens(res)
    assert t["삼아"]["alt"] == "살아" and t["2022"]["alt"] == "2023"
    assert t["2022"]["score"] >= 0.75                      # 숫자 불일치는 더 무겁게
    cands = cd.gen_candidates(t["삼아"], v)
    assert cands[0]["text"] == "살아" and set(cands[0]["sources"]) >= {"vocab", "alt_ocr"}   # 독립 출처 2개
    assert cands[0]["score"] > cd.gen_candidates({**t["삼아"], "alt": None}, v)[0]["score"]


def test_page_number_notes():
    pages = [{"page": i, "page_number": i + 10} for i in range(1, 8)]
    pages[3]["page_number"] = 99
    notes = dt.page_number_notes(pages)
    assert [n["page"] for n in notes] == [4] and notes[0]["expected"] == 14


def test_protected_terms_not_flagged():
    pages = [{"page": 1, "text": "삼성전자는 좋다 삼성전자 삼성전자"}, {"page": 2, "text": "삼성전사는 없다"}]
    v = cd.build_vocab([p["text"] for p in pages], P)
    assert "삼성전사는" in {s["token"] for s in dt.detect_pages(pages, v, P)["spans"]}      # 오탈자는 잡힘
    pr = nz.Protect(["삼성전사"])      # 어간만 등록해도 조사 붙은 형태가 보호됨
    assert "삼성전사는" not in {s["token"] for s in dt.detect_pages(pages, v, pr)["spans"]}  # 용어집에 있으면 보호
