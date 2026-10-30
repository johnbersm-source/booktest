"""tesseract 모드·--pages·embed·429 구분 테스트 (합성 데이터). tesseract 가 없으면 해당 테스트는 건너뜀."""
import json
import sys
from pathlib import Path

import pymupdf as fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_scan  # noqa: E402
from bookpdf import cli, ocr, ocr_tesseract, pagesel, pipeline, structure  # noqa: E402

needs_tess = pytest.mark.skipif(ocr_tesseract.check_available("kor+eng") is not None, reason="tesseract(kor) 없음")


# ---------- tesseract 없이 도는 단위 테스트 ----------
def test_parse_pages():
    assert pagesel.parse_pages("114-116") == [114, 115, 116]
    assert pagesel.parse_pages("1-3,10,2") == [1, 2, 3, 10]
    for bad in ("", "a", "5-3", "0-2", "1-"):
        with pytest.raises(ValueError):
            pagesel.parse_pages(bad)
    with pytest.raises(ValueError):
        pagesel.parse_pages("1-300", total=215)


class E(Exception):
    def __init__(self, msg, code=429):
        super().__init__(msg)
        self.code = code


def test_classify_quota():
    assert ocr.classify_quota_error(E("RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")) == "daily"
    assert ocr.classify_quota_error(E("RESOURCE_EXHAUSTED GenerateRequestsPerMinutePerProjectPerModel retry in 20s")) == "minute"
    assert ocr.classify_quota_error(E("429 RESOURCE_EXHAUSTED")) == "unknown"
    assert ocr.classify_quota_error(E("boom", code=500)) is None


def blk(text, zone="body", lh=40, lines=1, chars=None, y=0):
    return {"text": text, "zone": zone, "lh": lh, "lines": lines, "chars": len(text) if chars is None else chars,
            "y0": y, "y1": y + lh, "x0": 0, "x1": 0, "conf": 90}


def test_heading_rules():
    assert structure.heading_level(blk("제1장 시작", lh=45), 40) == 1       # 장 패턴은 1.1배만 넘어도 대제목
    assert structure.heading_level(blk("큰 제목", lh=65), 40) == 1
    assert structure.heading_level(blk("작은 제목", lh=52), 40) == 2
    assert structure.heading_level(blk("2.1 절 제목", lh=45), 40) == 2
    assert structure.heading_level(blk("큰 글자지만 문장으로 끝난다.", lh=65), 40) == 0
    assert structure.heading_level(blk("본문 크기 줄", lh=40), 40) == 0
    assert structure.heading_level(blk("머리글", zone="top", lh=65), 40) == 0


def test_repeated_edge_and_assemble(tmp_path):
    img = tmp_path / "p.png"
    from PIL import Image
    Image.new("RGB", (100, 100), "white").save(img)
    res = []
    for p in range(1, 5):
        res.append({"page": p, "image": str(img), "chars": 200, "ink": 0.0, "page_number": 10 + p if p != 3 else None,
                    "blocks": [blk("책 제목 머리글", "top", 30, y=10), blk("본문 " * 20, lh=40, y=900), blk("장 제목", lh=70, y=300)]})
    mds, figs, toc, st = structure.assemble(res, structure.EmbedOptions(), tmp_path / "e")
    assert st["edge_removed"] == 4 and all("머리글" not in m for m in mds)
    assert [t["text"] for t in toc] == ["장 제목"] * 4
    assert toc[2]["printed"] == 13 and toc[2]["est"] is True and toc[0]["est"] is False   # 3쪽 번호는 앞뒤에서 보정


def test_embed_decision(tmp_path):
    from PIL import Image
    img = tmp_path / "p.png"
    Image.new("RGB", (200, 300), "white").save(img)
    r = {"page": 1, "image": str(img), "chars": 10, "ink": 0.2, "blocks": [blk("그림 1 매출", lh=40, chars=8)]}
    off = structure.assemble([r], structure.EmbedOptions(False), tmp_path / "e")
    on = structure.assemble([r], structure.EmbedOptions(True, 120, 0.04), tmp_path / "e")
    assert "{{FIG:" not in off[0][0] and "{{FIG:full1}}" in on[0][0] and "full1" in on[1]
    blank = dict(r, ink=0.0)
    assert "{{FIG:" not in structure.assemble([blank], structure.EmbedOptions(True), tmp_path / "e")[0][0]   # 빈 쪽은 삽입 안 함
    texty = dict(r, chars=900)
    assert "{{FIG:" not in structure.assemble([texty], structure.EmbedOptions(True), tmp_path / "e")[0][0]


def test_quota_hint_and_stop(tmp_path):
    from PIL import Image
    imgs = []
    for i in range(1, 5):
        p = tmp_path / f"p{i}.png"
        Image.new("RGB", (50, 50), "white").save(p)
        imgs.append(p)
    n = {"c": 0}

    def fake(_p):
        n["c"] += 1
        if n["c"] == 3:
            raise ocr.QuotaExhausted("일일 한도(RPD) 도달: x", "daily")
        return {"markdown": "글", "figures": [], "confidence": 0.9, "page_number": None, "engine": "gemini"}

    logs = []
    res, stop = pipeline.process_pages(imgs, tmp_path / "w", fake, log=logs.append)
    assert stop and len(res) == 2
    hint = pipeline.read_quota_hint(tmp_path / "w")
    assert hint["kind"] == "daily" and hint["successful_calls"] == 2
    assert any("일일 한도(RPD)" in l for l in logs)


# ---------- tesseract 실제 실행 ----------
@pytest.fixture(scope="module")
def scan_pdf(tmp_path_factory):
    d = tmp_path_factory.mktemp("scan")
    make_scan.build_scan_pdf(d / "scan.pdf")
    return d / "scan.pdf"


def run(argv):
    return cli.main([str(a) for a in argv])


@needs_tess
def test_full_book_tesseract(scan_pdf, tmp_path, capsys):
    out = tmp_path / "out"
    rc = run([scan_pdf, "--ocr", "tesseract", "--dpi", "300", "--embed-page-images", "--out", out, "--name", "b.pdf"])
    cap = capsys.readouterr().out
    assert rc in (0, 4), cap
    md = (out / "전체원문.md").read_text(encoding="utf-8")
    assert "시험용 바다 이야기" not in md                 # 반복 머리글 제거
    assert not any(l.strip() in {"11", "12", "13", "14", "15", "16"} for l in md.splitlines())   # 쪽번호 제거
    assert "{{FIG:full3}}" in md                            # 3쪽(차트)은 원본 이미지 삽입
    assert "# 제" in md                                     # 제목 판정
    doc = fitz.open(out / "b.pdf")
    text = "".join(p.get_text() for p in doc)
    assert "목차" in text
    assert sum(len(p.get_images()) for p in doc) >= 1
    rep = (out / "리포트.md").read_text(encoding="utf-8")
    assert "tesseract" in rep and "picture_pages" in rep
    pn = [json.loads(p.read_text(encoding="utf-8"))["page_number"] for p in sorted((out / "work" / "pages").glob("p*.json"))]
    got = {i + 1: n for i, n in enumerate(pn) if n is not None}
    assert len(got) >= 4 and all(n == 10 + i for i, n in got.items())     # 읽힌 번호는 맞아야 하고, OCR 이 놓친 쪽(None)은 일부 허용


@needs_tess
def test_pages_range_and_resume(scan_pdf, tmp_path, capsys):
    out = tmp_path / "out"
    run([scan_pdf, "--ocr", "tesseract", "--dpi", "300", "--pages", "2-4", "--out", out])
    assert len(list((out / "work" / "pages").glob("p*.json"))) == 3
    capsys.readouterr()
    run([scan_pdf, "--ocr", "tesseract", "--dpi", "300", "--pages", "2-4", "--out", out])
    assert "처리 대상" not in capsys.readouterr().out          # 전부 완료돼 있으면 다시 처리하지 않는다
    # 설정(범위)을 바꿔 이어하기 → 거부(exit 2)
    assert run([scan_pdf, "--ocr", "tesseract", "--dpi", "300", "--pages", "1-4", "--out", out]) == 2
    # 범위 초과
    assert run([scan_pdf, "--ocr", "tesseract", "--pages", "1-99", "--out", tmp_path / "o2"]) == 2


@needs_tess
def test_engine_change_refused(scan_pdf, tmp_path):
    out = tmp_path / "out"
    run([scan_pdf, "--ocr", "tesseract", "--dpi", "300", "--pages", "1-2", "--out", out])
    assert run([scan_pdf, "--ocr", "gemini", "--dpi", "300", "--pages", "1-2", "--out", out]) == 2


# ---------- Gemini 429: 분당/일일 구분 (가짜 클라이언트) ----------
def _fake_gemini(monkeypatch, errors):
    genai = pytest.importorskip("google.genai")
    calls = {"n": 0}

    class Models:
        def generate_content(self, **kw):
            calls["n"] += 1
            e = errors[min(calls["n"] - 1, len(errors) - 1)]
            if isinstance(e, Exception):
                raise e

            class R:
                text = json.dumps({"markdown": "글", "figures": [], "confidence": 0.9})
            return R()

    class Client:
        def __init__(self, **kw):
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    monkeypatch.setattr(ocr.time, "sleep", lambda s: None)
    return calls


def test_gemini_daily_429_stops_without_retry(monkeypatch, tmp_path):
    calls = _fake_gemini(monkeypatch, [E("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    f = ocr.make_gemini_ocr("k", "m", 0, log=lambda s: None)
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    with pytest.raises(ocr.QuotaExhausted) as ei:
        f(img)
    assert ei.value.kind == "daily" and calls["n"] == 1


def test_gemini_minute_429_retries_then_succeeds(monkeypatch, tmp_path):
    calls = _fake_gemini(monkeypatch, [E("429 RESOURCE_EXHAUSTED GenerateRequestsPerMinutePerProjectPerModel retry in 3s"), None])
    logs = []
    f = ocr.make_gemini_ocr("k", "m", 0, log=logs.append)
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    assert f(img)["markdown"] == "글" and calls["n"] == 2
    assert any("분당 한도(RPM)" in l for l in logs)


def test_gemini_persistent_minute_429_becomes_quota_stop(monkeypatch, tmp_path):
    _fake_gemini(monkeypatch, [E("429 RESOURCE_EXHAUSTED PerMinute retry in 1s")])
    f = ocr.make_gemini_ocr("k", "m", 0, max_attempts=3, log=lambda s: None)
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    with pytest.raises(ocr.QuotaExhausted) as ei:
        f(img)
    assert ei.value.kind == "minute" and "GEMINI_RPM" in str(ei.value)


def test_cli_prints_max_calls_hint(scan_pdf, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    (out / "work").mkdir(parents=True)
    pipeline.write_quota_hint(out / "work", "daily", 7, "RequestsPerDay")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setattr(ocr, "make_gemini_ocr", lambda *a, **k: (lambda p: {}))
    seen = {}

    def fake_process(images, work, o, **kw):
        seen.update(kw)
        return [], "stub"

    monkeypatch.setattr(pipeline, "process_pages", fake_process)
    rc = run([scan_pdf, "--pages", "1-2", "--dpi", "100", "--out", out])
    txt = capsys.readouterr().out
    assert rc == 5 and "일일 한도(RPD)" in txt and "--max-calls 7" in txt and "태평양" in txt
    rc = run([scan_pdf, "--pages", "1-2", "--dpi", "100", "--max-calls", "5", "--out", out])
    assert seen["max_calls"] == 5          # 사용자가 준 상한이 그대로 전달된다
