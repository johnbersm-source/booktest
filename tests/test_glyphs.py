"""글꼴에 없는 글자(한자·원문자) 처리 테스트: 단위 + PDF 입력(텍스트 레이어에 날것의 ❶/兵 포함) 종단 검증."""
import sys
from pathlib import Path

import numpy as np
import pymupdf as fitz
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookpdf import cli, glyphs, render  # noqa: E402
from bookpdf.config import FONT_REGULAR  # noqa: E402


def test_classify():
    assert glyphs.classify("가") == "ok" and glyphs.classify("A") == "ok" and glyphs.classify(" ") == "ignore"
    assert glyphs.classify("兵") == "cjk" and glyphs.classify("三") == "cjk"
    assert glyphs.classify("❶") == "circled" and glyphs.classify("①") == "circled" and glyphs.classify("㉠") == "circled"
    assert glyphs.classify("⏣") == "missing"          # 어느 글꼴에도 없음


def test_circled_replacement():
    assert glyphs.normalize("❶❷❸❹ ❿ ① ⑳ ㉠ ㉮ ㈜ Ⓐ ⓑ") == "(1)(2)(3)(4) (10) (1) (20) (가) (가) (주) (A) (b)"
    assert glyphs.normalize("⏣") == glyphs.PLACEHOLDER


def test_wrap_only_text_nodes():
    out = glyphs.wrap_cjk("<p class='兵'>가 兵赤 나</p><img src='兵.png'/>")
    assert out == "<p class='兵'>가 <span class=\"cjk\">兵赤</span> 나</p><img src='兵.png'/>"
    assert glyphs.wrap_cjk("한글만") == "한글만"


def test_scan_and_report():
    found = glyphs.scan([(1, "兵赤 ❶ ⏣ 가나"), (2, "兵 ❶ {{FIG:p2_1}}")])
    assert found["cjk"]["兵"] == [2, [1, 2]] and found["circled"]["❶"][0] == 2 and "⏣" in found["missing"]
    text = "\n".join(glyphs.format_report(found))
    assert "U+5175" in text and "❶ → (1)" in text and "어느 글꼴에도 없음" in text
    assert "없음(모든 글자가" in "\n".join(glyphs.format_report(glyphs.scan([(1, "가나다 abc")])))


def _make_input_pdf(path: Path) -> None:
    """텍스트 레이어에 날것의 한자·원문자가 들어 있는 입력 PDF(여러 쪽, 글꼴은 번들 Noto/Nanum 으로 직접 삽입)."""
    doc = fitz.open()
    for p in range(3):
        pg = doc.new_page()
        pg.insert_font(fontname="nan", fontfile=str(FONT_REGULAR))
        pg.insert_font(fontname="cjk", fontfile=str(glyphs.CJK_REGULAR))
        pg.insert_text((72, 100), f"{p + 1}번째 쪽: 다음 글자는 원문에 있던 것이다.", fontname="nan", fontsize=12)
        pg.insert_text((72, 130), "兵赤黑三 ❶❷❸❹ ⑩ ㉠", fontname="cjk", fontsize=12)
        pg.insert_text((72, 160), "끝에 같은 문장이 반복된다. " * 3, fontname="nan", fontsize=12)
    doc.save(path)


def test_end_to_end_pdf(tmp_path):
    src = tmp_path / "in.pdf"
    _make_input_pdf(src)
    assert "兵赤黑三" in fitz.open(src)[0].get_text()           # 입력의 텍스트 레이어가 날것인지 확인
    out = tmp_path / "out"
    rc = cli.main([str(src), "--skip-ocr-if-text", "--out", str(out), "--name", "b.pdf"])
    assert rc == 0
    doc = fitz.open(out / "b.pdf")
    text = "".join(p.get_text() for p in doc)
    assert "兵赤黑三" in text                                    # 한자는 그대로 출력
    assert "(1)(2)(3)(4)" in text and "(10)" in text and "(가)" in text
    assert not any(c in text for c in "❶❷❸❹➊➋➌➍⑩➉㉠")
    fonts = {f[3] for p in doc for f in p.get_fonts()}
    assert any("NotoSansCJK" in f for f in fonts) and any("Nanum" in f for f in fonts)
    rep = (out / "리포트.md").read_text(encoding="utf-8")
    # 입력 PDF 의 텍스트 추출이 ❶ 을 모양이 같은 ➊(U+278A)로 돌려줄 수 있어 코드포인트가 아니라 치환 결과로 확인
    assert "글꼴에 없는 글자" in rep and "兵 (U+5175)" in rep and "→ (1) (U+" in rep and "→ (가) (U+3260)" in rep
    # 글리프가 실제로 그려졌는가: 한자 줄의 잉크량이 □ 대체 줄과 비슷한 '빈칸'이 아니어야 한다
    page = [p for p in doc if "兵赤黑三" in p.get_text()][0]
    r = page.search_for("兵赤黑三")[0]
    pix = page.get_pixmap(clip=r, dpi=150)
    gray = np.asarray(Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L"))
    dark = float((gray < 128).mean())          # 대조 실험: 글리프 있는 Noto 0.165, 글리프 없는 Nanum 0.0
    assert dark > 0.08, f"한자 영역 잉크 비율 {dark:.3f}: 빈칸 가능성"


def test_cjk_font_missing_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(glyphs, "CJK_REGULAR", tmp_path / "nope.ttf")
    glyphs._cmaps.cache_clear()
    try:
        assert glyphs.classify("兵") == "missing"
        assert "경고" in "\n".join(glyphs.format_report(glyphs.scan([(1, "兵")])))
    finally:
        glyphs._cmaps.cache_clear()
