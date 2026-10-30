"""펼침면·긴 영상·한도 중단/이어하기·쪽번호 점검·설정 서명 테스트 (Gemini 는 가짜)."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import make_spread as ms  # noqa: E402
from bookpdf import cli, pipeline, report  # noqa: E402
from bookpdf.ocr import QuotaExhausted  # noqa: E402


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr("bookpdf.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_RPD", raising=False)


def _spread_pdf(path: Path, n: int = 3) -> None:
    doc = pymupdf.open()
    for k in range(n):
        img, _ = ms.make_spread(seed=10 + k)
        buf = path.parent / f"sp{k}.png"
        img.save(buf)
        page = doc.new_page(width=img.width * 0.5, height=img.height * 0.5)  # 가로 펼침면 비율
        page.insert_image(page.rect, filename=str(buf))
    doc.save(path)


def _side_ocr(calls: list):
    """좌/우 쪽을 색 표식으로 구분해 '왼쪽N/오른쪽N' 본문 + 쪽번호(2k-1, 2k)를 돌려주는 가짜 OCR."""
    def fake(img: Path) -> dict:
        im = Image.open(img)
        red, blue = ms.count_color(im, "red"), ms.count_color(im, "blue")
        n = len(calls) + 1
        calls.append(img.name)
        side = "왼쪽" if red > blue else "오른쪽"
        return {"markdown": f"## {side}{n}\n\n{side} 본문 {n} 가나다라마바사아자차", "figures": [],
                "confidence": 0.9, "page_number": n, "engine": "gemini"}
    return fake


def test_pdf_spreads_split_in_left_to_right_order(tmp_path, monkeypatch):
    pdf = tmp_path / "spreads.pdf"
    _spread_pdf(pdf, 3)
    calls: list = []
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: _side_ocr(calls))
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    assert cli.main([str(pdf), "--out", str(out), "--title", "펼침면 책"]) == 0
    assert len(calls) == 6, "펼침면 3장은 6쪽으로 분리되어야 함"
    body = (out / "전체원문.md").read_text(encoding="utf-8")
    order = [body.index(f"## {s}{i}") for i, s in enumerate(["왼쪽", "오른쪽"] * 3, start=1)]
    assert order == sorted(order), "좌→우 순서가 아님"
    rep = (out / "리포트.md").read_text(encoding="utf-8")
    assert "원본 3장 → 책 6쪽 (펼침면 3장 분리)" in rep
    assert "누락·중복·역행은 발견되지 않았습니다" in rep
    assert len(pymupdf.open(out / "책.pdf")) >= 3


def test_spread_off_keeps_pages_whole(tmp_path, monkeypatch):
    pdf = tmp_path / "spreads.pdf"
    _spread_pdf(pdf, 2)
    calls: list = []
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: _side_ocr(calls))
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(pdf), "--out", str(tmp_path / "o"), "--spread", "off"]) == 0
    assert len(calls) == 2


def test_quota_stop_then_resume_and_no_book_when_partial(tmp_path, monkeypatch):
    pdf = tmp_path / "spreads.pdf"
    _spread_pdf(pdf, 3)  # 6쪽
    state = {"n": 0, "limit": 4}
    seen: list = []

    def fake(img: Path) -> dict:
        if state["n"] >= state["limit"]:
            raise QuotaExhausted("일일 한도 도달: RequestsPerDay")
        state["n"] += 1
        seen.append(img.name)
        k = int(img.stem[1:])
        return {"markdown": f"## {k}쪽\n\n본문 {k} 가나다라마바사아자차", "figures": [], "confidence": 0.9,
                "page_number": k, "engine": "gemini"}

    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: fake)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    assert cli.main([str(pdf), "--out", str(out)]) == 5           # 한도 도달: 책은 만들지 않음
    assert not (out / "책.pdf").exists()
    assert len(list((out / "work/pages").glob("p*.json"))) == 4   # 처리한 4쪽은 저장됨
    assert "미처리 2쪽" in (out / "리포트.md").read_text(encoding="utf-8")

    state["limit"] = 99                                          # 다음 날: 한도 회복
    seen.clear()
    assert cli.main([str(pdf), "--out", str(out)]) == 0
    assert seen == ["p0005.png", "p0006.png"], "남은 쪽만 처리해야 함"
    assert (out / "책.pdf").exists()


def test_allow_partial_builds_book_from_done_pages(tmp_path, monkeypatch):
    pdf = tmp_path / "spreads.pdf"
    _spread_pdf(pdf, 3)
    n = {"c": 0}

    def fake(img: Path) -> dict:
        if n["c"] >= 3:
            raise QuotaExhausted("per day")
        n["c"] += 1
        return {"markdown": f"## {n['c']}쪽\n\n본문 가나다라마바사아자차카타", "figures": [], "confidence": 0.9, "engine": "gemini"}

    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: fake)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(pdf), "--out", str(tmp_path / "o"), "--allow-partial"]) in (0, 4)
    assert (tmp_path / "o/책.pdf").exists()


def test_max_calls_stops_cleanly(tmp_path, monkeypatch):
    pdf = tmp_path / "spreads.pdf"
    _spread_pdf(pdf, 3)
    calls: list = []
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: _side_ocr(calls))
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(pdf), "--out", str(tmp_path / "o"), "--max-calls", "2"]) == 5
    assert len(calls) == 2
    monkeypatch.setenv("GEMINI_RPD", "3")        # 환경변수 일일 상한도 같은 동작
    assert cli.main([str(pdf), "--out", str(tmp_path / "o")]) == 5
    assert len(calls) == 5


def test_settings_change_blocked_until_restart(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "spreads.pdf"
    _spread_pdf(pdf, 2)
    calls: list = []
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: _side_ocr(calls))
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    out = str(tmp_path / "o")
    assert cli.main([str(pdf), "--out", out]) == 0
    n = len(calls)
    assert cli.main([str(pdf), "--out", out, "--spread", "off"]) == 2   # 설정이 달라 이전 결과와 섞이면 안 됨
    assert "spread" in capsys.readouterr().err and len(calls) == n
    assert cli.main([str(pdf), "--out", out, "--spread", "off", "--restart"]) == 0
    assert len(calls) == n + 2


def test_page_number_gap_and_duplicate_in_report(tmp_path, monkeypatch):
    pdf = tmp_path / "spreads.pdf"
    _spread_pdf(pdf, 4)  # 8쪽
    nums = [1, 2, 3, 5, 6, 6, 8, 9]   # 4 누락, 6 중복
    it = iter(nums)

    def fake(img: Path) -> dict:
        k = int(img.stem[1:])
        return {"markdown": f"## {k}쪽\n\n본문 {k} 가나다라마바사아자차", "figures": [], "confidence": 0.9,
                "page_number": next(it), "engine": "gemini"}

    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: fake)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(pdf), "--out", str(tmp_path / "o")]) == 0
    rep = (tmp_path / "o/리포트.md").read_text(encoding="utf-8")
    assert "누락 의심" in rep and "번호 4" in rep
    assert "중복 의심" in rep and "같은 번호(6)" in rep


def test_eta_progress_message():
    t = {"now": 0.0}
    logs: list = []

    def clock():
        return t["now"]

    def ocr(img):
        t["now"] += 6.0
        return {"markdown": "x" * 30, "figures": [], "confidence": 0.9, "engine": "gemini"}

    import tempfile
    d = Path(tempfile.mkdtemp())
    imgs = []
    for i in range(1, 5):
        p = d / f"p{i:04d}.png"
        Image.new("RGB", (60, 60), "white").save(p)
        imgs.append(p)
    pipeline.process_pages(imgs, d / "work", ocr, log=logs.append, rpm=10, clock=clock)
    assert any("최소 약 24초" in m for m in logs[:1]), logs[:1]            # 4쪽 × 60/10 = 24초
    assert any("쪽당 평균 6.0초" in m and "남은 3쪽 약 18초" in m for m in logs), logs


def test_check_page_numbers_ignores_unnumbered_pages():
    mk = lambda p, n: {"page": p, "page_number": n, "markdown": "x"}  # noqa: E731
    res = [mk(1, None), mk(2, 1), mk(3, 2), mk(4, None), mk(5, 4), mk(6, 5)]
    chk = report.check_page_numbers(res)
    assert chk["missing"] == [] and chk["duplicates"] == [] and chk["reversed"] == []
    assert report.check_page_numbers([mk(1, 1), mk(2, 2)])["usable"] is False


# ---------------- 영상 (로컬 전용; ffmpeg + opencv 필요) ----------------
cv2 = pytest.importorskip("cv2")
pytestmark_video = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg 없음")


def _enc(codec="libx264"):
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return codec in out


@pytestmark_video
def test_video_spreads_end_to_end(tmp_path, monkeypatch):
    import make_video
    spreads = []
    for k in range(3):
        im, _ = ms.make_spread(seed=20 + k)
        spreads.append(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
    mov = tmp_path / "spreads.mov"
    make_video.build_from_images(spreads, mov, codec="libx264", rotation=0, hold_s=2.0, fps=30)
    calls: list = []
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: _side_ocr(calls))
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    assert cli.main([str(mov), "--out", str(out), "--fps", "3"]) == 0
    assert len(calls) == 6, f"펼침면 3장→6쪽이어야 함: {calls}"
    body = (out / "전체원문.md").read_text(encoding="utf-8")
    idx = [body.index(f"## {s}{i}") for i, s in enumerate(["왼쪽", "오른쪽"] * 3, start=1)]
    assert idx == sorted(idx)
    # 스트리밍: 전체 프레임을 디스크에 풀어놓지 않는다
    assert not (out / "work" / "frames").exists()
    assert len(list((out / "work/source_images").glob("p*.jpg"))) == 3


@pytestmark_video
def test_long_video_30_pages_streaming_and_resume(tmp_path, monkeypatch):
    import make_video
    rng = np.random.default_rng(5)
    pages = []
    for k in range(30):
        img = np.full((480, 360, 3), 245, np.uint8)
        for row in range(12):
            txt = "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ23456789"), 9))
            cv2.putText(img, txt, (20, 40 + row * 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
        pages.append(img)
    mov = tmp_path / "long.mov"
    make_video.build_from_images(pages, mov, codec="libx264", rotation=0, hold_s=1.0, fps=10, transition=3)
    from bookpdf.local import video_input
    out, info = video_input.extract_pages(mov, tmp_path / "w" / "src", fps=3, log=lambda *_: None)
    assert len(out) == 30, info
    assert not (tmp_path / "w" / "frames").exists()
    # 같은 영상·설정이면 재분석하지 않는다
    m = out[0].stat().st_mtime_ns
    out2, _ = video_input.extract_pages(mov, tmp_path / "w" / "src", fps=3, log=lambda *_: None)
    assert [p.name for p in out2] == [p.name for p in out] and out2[0].stat().st_mtime_ns == m
    # 순서 확인: 각 결과가 원본 순서와 같은 쪽
    for k in (0, 7, 29):
        g = cv2.cvtColor(cv2.imread(str(out[k])), cv2.COLOR_BGR2GRAY)
        diffs = [video_input.page_diff(video_input._small(g), video_input._small(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY))) for p in pages]
        assert int(np.argmin(diffs)) == k, f"쪽 {k + 1} 순서/내용 불일치"
