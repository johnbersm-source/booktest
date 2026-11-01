"""--drop-junk: 신뢰도 낮고 글자 적은 쪽을 책 PDF 에서 제외하고 리포트에 목록을 남긴다."""
import sys
from pathlib import Path

import pymupdf as fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_scan  # noqa: E402
from bookpdf import cli, junk, ocr_tesseract  # noqa: E402

needs_tess = pytest.mark.skipif(ocr_tesseract.check_available("kor+eng") is not None, reason="tesseract(kor) 없음")


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr("bookpdf.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_RPD", raising=False)


def R(page, conf, **kw):
    return {"page": page, "confidence": conf, "markdown": "", **kw}


# ---------- 단위 ----------
def test_requires_both_low_confidence_and_few_chars():
    md = {1: "가" * 10, 2: "가" * 10, 3: "가" * 100, 4: "가" * 100, 5: "", 6: "가" * 10}
    res = [R(1, 0.2), R(2, 0.35), R(3, 0.2), R(4, 0.9), R(5, 0.0), R(6, 0.34)]
    got = junk.find_junk(res, md, 0.35, 30)
    assert [d["page"] for d in got] == [1, 5, 6]               # 2: 신뢰도가 기준과 같음(미만 아님), 3: 글자 많음, 4: 신뢰도 높음
    assert got[0]["chars"] == 10 and "0.20 < 0.35" in got[0]["reason"] and "10자 < 30자" in got[0]["reason"]


def test_protected_pages_are_never_dropped():
    md = {p: "가" for p in range(1, 7)}
    res = [R(1, 0.1, embedded=True), R(2, 0.1, error="x"), R(3, 0.1, figures=[{"box_2d": [0, 0, 9, 9]}]),
           R(4, 0.1), R(5, 0.1), R(6, 0.1)]
    got = junk.find_junk(res, md, 0.35, 30, protect_pages={5})
    assert [d["page"] for d in got] == [4, 6]


def test_char_count_ignores_spaces_and_figure_tokens_and_uses_assembled_text():
    assert junk.count_chars("가 나\n다 {{FIG:full3}} ") == 3
    assert junk.count_chars("") == 0 and junk.count_chars(None) == 0


def test_report_lines_lists_pages_reasons_and_warns_when_many():
    d = [{"page": 4, "printed": 259, "confidence": 0.12, "chars": 3, "reason": "신뢰도 0.12 < 0.35 이고 글자 3자 < 30자", "preview": "a|b"}]
    txt = "\n".join(junk.report_lines(d, 10, 0.35, 30))
    assert "| p4 | 259 | 0.12 | 3 | 신뢰도 0.12 < 0.35 이고 글자 3자 < 30자 | a\\|b |" in txt
    assert "제외 1쪽 / 전체 10쪽 (10%)" in txt and "경고" not in txt
    assert "경고" in "\n".join(junk.report_lines(d * 3, 10, 0.35, 30))
    assert "제외한 쪽 없음" in "\n".join(junk.report_lines([], 10, 0.35, 30))


# ---------- CLI (Gemini 가짜) ----------
def _pdf(path: Path, n: int) -> None:
    doc = fitz.open()
    for _ in range(n):
        doc.new_page()
    doc.save(path)


PAGES = {  # 쪽 → (본문, 신뢰도, 그림 여부)
    1: ("# 첫 장\n\n정상 본문입니다. 충분히 긴 문장이 여기에 들어 있습니다. 가나다라마바사아자차카타파하.", 0.9, False),
    2: ("ㅁ", 0.10, False),                                   # 쓰레기: 낮은 신뢰도 + 글자 거의 없음 → 제외
    3: ("긴 본문이지만 신뢰도가 낮은 쪽입니다. 글자가 충분히 많아서 제외되면 안 됩니다. 가나다라마바사아자차카타.", 0.20, False),
    4: ("짧음", 0.90, False),                                  # 신뢰도 높음 → 유지
    5: ("", 0.0, False),                                      # 빈 쪽 + 신뢰도 0 → 제외
    6: ("표", 0.10, True),                                    # 그림 영역 있음 → 유지
    7: ("끝 본문입니다. 마지막 쪽의 충분히 긴 문장입니다. 가나다라마바사아자차카타파하.", 0.9, False),
}


def _fake_gemini(monkeypatch):
    def maker(*a, **k):
        def f(img: Path) -> dict:
            n = int(Path(img).stem.lstrip("p"))
            md, conf, fig = PAGES[n]
            return {"markdown": md, "confidence": conf, "page_number": n, "engine": "gemini",
                    "figures": [{"box_2d": [100, 100, 800, 800], "caption": "표"}] if fig else []}
        return f
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", maker)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")


def test_cli_drop_junk_excludes_pages_and_reports_them(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "b.pdf"
    _pdf(pdf, 7)
    _fake_gemini(monkeypatch)
    out = tmp_path / "out"
    rc = cli.main([str(pdf), "--out", str(out), "--drop-junk", "--dpi", "80"])
    cap = capsys.readouterr().out
    assert rc in (0, 4), cap
    assert "--drop-junk: 2쪽 제외 (p2, p5)" in cap
    body = (out / "전체원문.md").read_text(encoding="utf-8")
    assert "ㅁ" not in body and "정상 본문" in body and "신뢰도가 낮은 쪽입니다" in body and "짧음" in body
    rep = (out / "리포트.md").read_text(encoding="utf-8")
    assert "## 책 PDF 에서 제외한 쪽 (--drop-junk)" in rep and "| p2 | 2 | 0.10 | 1 |" in rep and "| p5 |" in rep
    # 원본 결과 보존
    assert (out / "work" / "pages" / "p0002.json").exists() and (out / "pages_md" / "p0002.md").read_text(encoding="utf-8") == "ㅁ"
    # '신뢰도 낮은 페이지'에는 제외하지 않은 p3(긴 본문, 낮은 신뢰도)·p6(그림 있음)만 남고 제외 쪽(p2, p5)은 빠짐
    low_section = rep.split("## 신뢰도 낮은 페이지")[1]
    assert "- p3:" in low_section and "- p6:" in low_section and "- p2:" not in low_section and "- p5:" not in low_section


def test_cli_without_flag_keeps_everything(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "b.pdf"
    _pdf(pdf, 7)
    _fake_gemini(monkeypatch)
    out = tmp_path / "out"
    assert cli.main([str(pdf), "--out", str(out), "--dpi", "80"]) in (0, 4)
    assert "ㅁ" in (out / "전체원문.md").read_text(encoding="utf-8")
    assert "제외한 쪽" not in (out / "리포트.md").read_text(encoding="utf-8")


def test_cli_rejects_bad_junk_values(tmp_path, capsys):
    pdf = tmp_path / "b.pdf"
    _pdf(pdf, 1)
    for bad in (["--junk-conf", "0"], ["--junk-conf", "1.2"], ["--junk-chars", "0"]):
        assert cli.main([str(pdf), "--out", str(tmp_path / "o"), "--drop-junk", *bad]) == 2
    assert "--junk-conf" in capsys.readouterr().err


@pytest.fixture(scope="module")
def scan_pdf(tmp_path_factory):
    d = tmp_path_factory.mktemp("scan")
    make_scan.build_scan_pdf(d / "scan.pdf")
    return d / "scan.pdf"


@needs_tess
def test_cli_drop_junk_never_drops_embedded_or_heading_pages_even_with_extreme_thresholds(scan_pdf, tmp_path, capsys):
    out = tmp_path / "out"
    rc = cli.main([str(scan_pdf), "--ocr", "tesseract", "--dpi", "300", "--embed-page-images", "--out", str(out), "--name", "b.pdf",
                   "--drop-junk", "--junk-conf", "1.0", "--junk-chars", "100000"])
    cap = capsys.readouterr().out
    assert rc in (0, 4), cap
    body = (out / "전체원문.md").read_text(encoding="utf-8")
    assert "{{FIG:full3}}" in body                              # 차트 쪽(원본 삽입)은 제외되지 않음
    assert "# 제" in body                                        # 제목이 있는 쪽도 제외되지 않음
    rep = (out / "리포트.md").read_text(encoding="utf-8")
    assert "p3 |" not in rep.split("## 책 PDF 에서 제외한 쪽")[1].split("\n## ")[0]
