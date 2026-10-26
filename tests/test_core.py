"""Gemini 경로는 가짜 클라이언트로 검증한다(실제 API 호출 없음)."""
import json
import sys
import types
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import make_sample  # noqa: E402
from bookpdf import cli, ocr, pdf_input, pipeline, postprocess, report  # noqa: E402


# ---------- ocr.parse_response ----------
def test_parse_response_ok_and_clamp():
    r = ocr.parse_response('```json\n{"markdown":"안녕","figures":[{"box_2d":[-5,10,2000,500],"caption":"c"},'
                           '{"box_2d":[10,10,10,50]},{"box_2d":"x"}],"confidence":1.7}\n```')
    assert r["markdown"] == "안녕"
    assert r["figures"] == [{"box_2d": [0, 10, 1000, 500], "caption": "c"}]  # 잘못된 좌표는 버림, 범위 보정
    assert r["confidence"] == 1.0


def test_parse_response_invalid():
    with pytest.raises(ValueError):
        ocr.parse_response('{"text":"x"}')
    with pytest.raises(json.JSONDecodeError):
        ocr.parse_response("not json")


# ---------- 호출 제한 재시도 ----------
class FakeAPIError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def _install_fake_genai(monkeypatch, behaviors):
    calls = {"n": 0}

    class Models:
        def generate_content(self, **kw):
            calls["n"] += 1
            b = behaviors.pop(0)
            if isinstance(b, Exception):
                raise b
            return types.SimpleNamespace(text=b)

    class Client:
        def __init__(self, api_key):
            self.models = Models()

    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = Client
    fake_types = types.ModuleType("google.genai.types")
    fake_types.Part = types.SimpleNamespace(from_bytes=lambda data, mime_type: ("img", mime_type))
    fake_types.GenerateContentConfig = lambda **kw: kw
    fake_genai.types = fake_types
    import google
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    sleeps = []
    monkeypatch.setattr(ocr.time, "sleep", lambda s: sleeps.append(s))
    return calls, sleeps


GOOD = json.dumps({"markdown": "# 제목\n\n본문", "figures": [], "confidence": 0.9})


def test_retry_on_429_then_success(monkeypatch, tmp_path):
    calls, sleeps = _install_fake_genai(
        monkeypatch, [FakeAPIError(429, "RESOURCE_EXHAUSTED retry in 7s"), "깨진응답", GOOD])
    img = tmp_path / "p.png"
    Image.new("RGB", (50, 50), "white").save(img)
    fn = ocr.make_gemini_ocr("k", "m", rpm=0, log=lambda *_: None)
    res = fn(img)
    assert res["markdown"].startswith("# 제목") and calls["n"] == 3
    assert any(abs(s - 8.0) < 0.01 for s in sleeps)  # 서버가 알려준 7초 + 1초


def test_non_retryable_error_raises(monkeypatch, tmp_path):
    _install_fake_genai(monkeypatch, [FakeAPIError(400, "INVALID_ARGUMENT bad key")])
    img = tmp_path / "p.png"
    Image.new("RGB", (50, 50), "white").save(img)
    fn = ocr.make_gemini_ocr("k", "m", rpm=0, log=lambda *_: None)
    with pytest.raises(FakeAPIError):
        fn(img)


def test_rate_limiter_spacing():
    t = {"now": 0.0}
    slept = []
    rl = ocr.RateLimiter(rpm=6, clock=lambda: t["now"], sleep=lambda s: (slept.append(s), t.__setitem__("now", t["now"] + s)))
    rl.wait(); rl.wait()
    assert slept == [10.0]


# ---------- 후처리 ----------
def test_postprocess_keeps_numbered_headings_drops_running_header():
    pages = [f"러닝헤더 책\n\n## {p}장. 제목\n\n{p}쪽 본문 문장입니다 가나다라마바사.\n\n- {p} -" for p in range(1, 6)]
    body, st = postprocess.postprocess(pages)
    assert "러닝헤더" not in body and "- 1 -" not in body
    for p in range(1, 6):
        assert f"## {p}장. 제목" in body


def test_postprocess_hyphen_and_cross_page_merge():
    p1 = "exam-\nple word 처리\n\n이 문장은 쪽 경계에서 끊기고"
    p2 = "이어지는 부분입니다.\n\n다음 문단입니다."
    body, _ = postprocess.postprocess([p1, p2])
    assert "example word 처리" in body
    assert "이 문장은 쪽 경계에서 끊기고 이어지는 부분입니다." in body


def test_postprocess_dedupes_overlapping_pages_but_keeps_short_repeats():
    dup = "영상에서 겹쳐 찍힌 충분히 긴 문단입니다 중복."
    p1 = f"첫 문단입니다 서로 다른 내용.\n\n{dup}"
    p2 = f"{dup}\n\n새로운 내용의 문단입니다.\n\n네."
    p3 = "네.\n\n마지막 문단입니다 완전히 다른 내용."
    body, st = postprocess.postprocess([p1, p2, p3])
    assert body.count(dup) == 1 and st["duplicates_removed"] == 1
    assert body.count("네.") == 2  # 짧은 문구(정상 반복 가능)는 지우지 않는다


# ---------- 입력 검증 ----------
def test_validate_pdf_errors(tmp_path):
    with pytest.raises(ValueError):
        pdf_input.validate_pdf(tmp_path / "none.pdf")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(ValueError):
        pdf_input.validate_pdf(bad)


def test_cli_rejects_unsupported_ext(tmp_path, capsys):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert cli.main([str(f), "--out", str(tmp_path / "o")]) == 2


# ---------- 파이프라인: 이어서 실행 + end-to-end(이미지 전용 PDF) ----------
def _image_only_pdf(src: Path, dst: Path) -> None:
    d = pymupdf.open(src)
    out = pymupdf.open()
    for p in d:
        pix = p.get_pixmap(dpi=100)
        page = out.new_page(width=p.rect.width, height=p.rect.height)
        page.insert_image(page.rect, pixmap=pix)
    out.save(dst)


def test_end_to_end_image_pdf_with_fake_gemini_and_resume(tmp_path, monkeypatch):
    src = tmp_path / "text.pdf"
    make_sample.build(src)
    scan = tmp_path / "scan.pdf"
    _image_only_pdf(src, scan)
    assert not any(pdf_input.has_text_layer(scan))  # 텍스트 레이어 없음(OCR 필수)

    calls = {"n": 0}
    fail_once = {"p0002.png": True}

    def fake_ocr(img: Path) -> dict:
        calls["n"] += 1
        n = int(img.stem[1:])
        if fail_once.pop(img.name, False):
            raise RuntimeError("일시 장애")
        md = f"## {n}장. 스캔 제목\n\n{n}쪽 스캔 본문 문장입니다 가나다라마바사아자차."
        figs = [{"box_2d": [300, 100, 450, 500], "caption": "그림 설명"}] if n == 3 else []
        if figs:
            md += "\n\n{{FIG:1}}"
        return {"markdown": md, "figures": figs, "confidence": 0.9 if n != 5 else 0.3, "engine": "gemini"}

    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: fake_ocr)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    rc = cli.main([str(scan), "--out", str(out), "--title", "스캔 책"])
    assert rc == 0, "두 번째 쪽이 실패해도 나머지로 책 PDF 는 만들어져야 함"
    assert calls["n"] == 6
    assert not (out / "work/pages/p0002.json").exists()  # 실패 쪽은 캐시하지 않음
    rep = (out / "리포트.md").read_text(encoding="utf-8")
    assert "p2:" in rep and "p5:" in rep  # 실패 쪽 + 신뢰도 낮은 쪽이 리포트에 나옴

    # 재실행: 이미 처리한 5쪽은 다시 호출하지 않고, 실패했던 2쪽만 호출
    calls["n"] = 0
    rc = cli.main([str(scan), "--out", str(out), "--title", "스캔 책"])
    assert rc == 0 and calls["n"] == 1
    assert (out / "work/pages/p0002.json").exists()
    assert "p2:" not in (out / "리포트.md").read_text(encoding="utf-8")

    doc = pymupdf.open(out / "책.pdf")
    text = "".join(p.get_text() for p in doc)
    assert "스캔 제목" in text and doc.page_count >= 3
    assert sum(len(p.get_images()) for p in doc) >= 1  # 크롭한 그림 포함
    assert any("Nanum" in f[3] for p in doc for f in p.get_fonts())


def test_no_key_image_pdf_reports_failure(tmp_path, monkeypatch):
    src = tmp_path / "text.pdf"
    make_sample.build(src)
    scan = tmp_path / "scan.pdf"
    _image_only_pdf(src, scan)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("bookpdf.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    rc = cli.main([str(scan), "--out", str(tmp_path / "o")])
    assert rc == 3  # 결과 검증: 텍스트가 없으면 빈 책을 만들지 않고 실패 보고
    assert "OCR 엔진 없음" in (tmp_path / "o/리포트.md").read_text(encoding="utf-8")
    assert not list((tmp_path / "o/work/pages").glob("p*.json")), "키 없이 만든 빈 결과를 캐시하면 안 됨"

    # 나중에 키를 넣고 재실행하면 모든 쪽을 OCR 해야 한다
    seen = []
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: (lambda img: (seen.append(img.name), {
        "markdown": f"## {len(seen)}장\n\n본문 {len(seen)} 가나다라마바사아", "figures": [], "confidence": 0.9, "engine": "gemini"})[1]))
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    assert cli.main([str(scan), "--out", str(tmp_path / "o")]) == 0
    assert len(seen) == 6


def test_textlayer_path_without_key(tmp_path, monkeypatch):
    src = tmp_path / "text.pdf"
    make_sample.build(src)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("bookpdf.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    rc = cli.main([str(src), "--out", str(tmp_path / "o"), "--skip-ocr-if-text"])
    assert rc == 0
    md = (tmp_path / "o/전체원문.md").read_text(encoding="utf-8")
    assert "| 사과 | 1,200 | 국내산 |" in md and "{{FIG:p3_1}}" in md and "## 3장" in md
