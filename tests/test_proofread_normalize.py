"""M1 정규화 테스트 (합성 데이터)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookpdf.proofread import normalize as nz  # noqa: E402

P = nz.Protect()


def n(s, protect=P):
    return nz.normalize_line(s, protect)[0]


def test_nfc_and_invisible():
    nfd = "한글"          # 한글 (분해형)
    assert n(nfd + "​") == "한글"
    assert n("a b　c") == "a b c"


def test_fullwidth_quotes_hyphen():
    assert n("２０２１년 ＫＯＳＰＩ，") == "2021년 KOSPI,"
    assert n("“증가”와 ‘감소’") == '"증가"와 \'감소\''
    assert n("−5%  ‑3 –7") == "-5% -3 –7"                 # en dash 는 그대로
    assert nz.normalize_line("“a”", P, quote_style="keep")[0] == "“a”"


def test_spaces():
    assert n("한글   사이\t공백  ") == "한글 사이 공백"
    assert n("다 . 라 , 마") == "다. 라, 마"
    assert n("( 주 ) 설명") == "(주) 설명"
    assert n("  - 목록   항목") == "  - 목록 항목"          # 들여쓰기 유지
    assert n("1 . 항목") == "1 . 항목"                      # 숫자 뒤 공백은 건드리지 않음
    assert n("| a   | b |") == "| a | b |"


def test_digit_confusion_only_in_numeric_context():
    assert n("2O21년 매출은 1O0억원") == "2021년 매출은 100억원"
    assert n("성장률은 l0% 이다") == "성장률은 10% 이다"
    assert n("5rn 구간, 3rnrn") == "5m 구간, 3mm"
    # 일반 단어·영문·약어 안의 O/l/I 는 건드리지 않는다
    assert n("Olympic Illinois lol RSI14 COVID19") == "Olympic Illinois lol RSI14 COVID19"
    assert n("Oi 1l O.5") == "Oi 1l O.5"                    # 숫자 증거가 약하면 보수적으로 둔다


def test_protect_glossary_and_acronym():
    assert n("1O0") == "100" and n("1O0", nz.Protect(["1O0"])) == "1O0"   # 용어집에 있으면 숫자 규칙도 적용 안 함
    pr = nz.Protect(["O2O", "I2C"])
    assert n("O2O와 I2C", pr) == "O2O와 I2C"
    assert "MACD" in nz.Protect() and "KOSPI" in nz.Protect()


def test_figure_marker_preserved():
    assert n("그림 {{FIG:p1_1}}  끝") == "그림 {{FIG:p1_1}} 끝"


def test_text_roundtrip_and_line_count():
    md = "# 제목\n\n２０２１년  “상승”\n| a | b |\n끝 ."
    r = nz.normalize_text(md, P)
    assert r.text.count("\n") == md.count("\n")             # 줄 수 불변
    assert {c["line"] for c in r.changes} == {3, 5}
    assert nz.revert_text(r.text, r.changes) == md           # 완전 복원
    assert nz.normalize_text(r.text, P).changes == []        # 멱등


def test_revert_detects_tampering():
    import pytest
    r = nz.normalize_text("２０２１", P)
    with pytest.raises(ValueError):
        nz.revert_text("다른 내용", r.changes)


def test_glossary_collect_and_files(tmp_path):
    texts = ["이동평균선은 중요하다 MACD MACD", "이동평균선이 상승 MACD", "이동평균선을 본다 MACD MACD", "한번만"]
    got = {w: (c, p) for w, c, p in nz.collect_glossary(texts, min_count=3, min_pages=2)}
    assert got["이동평균선"] == (3, 3) and got["MACD"] == (5, 3) and "한번만" not in got
    f = tmp_path / "glossary_auto.txt"
    nz.write_glossary(f, [("이동평균선", 3, 3)])
    g = tmp_path / "glossary.txt"
    g.write_text("# 주석\n삼성전자\t12\n\n한국거래소\n", encoding="utf-8")
    pr = nz.Protect.load(glossary=g, auto=f)
    assert "이동평균선" in pr and "삼성전자" in pr and "한국거래소" in pr and "주석" not in pr


def test_fullwidth_rule_never_produces_unrenderable_glyphs():
    """전각 규칙이 만든 글자가 번들 글꼴(NanumGothic/Noto)에 없어 PDF 에서 □ 가 되는 일이 없어야 한다."""
    from bookpdf import glyphs
    for cp in range(0xFF00, 0xFFF0):
        out = n(chr(cp))
        if out == chr(cp):
            continue                                       # 규칙이 건드리지 않은 글자는 원래 상태 그대로
        for ch in out:
            if ch.isspace():
                continue
            assert glyphs.classify(ch) in ("ok", "cjk", "circled", "ignore"), (hex(cp), repr(out))
    assert n("ﾡ") == "ﾡ" and n("￣") == "￣"               # 반각 한글 자모/￣ 는 그대로 둠
    assert n("￥１００ ￦５００") == "¥100 ₩500"
