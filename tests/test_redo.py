"""--redo-below: 신뢰도 낮은 쪽만 다른 모델로 재처리 (합성 데이터, Gemini 는 가짜)."""
import json
import sys
from pathlib import Path

import pymupdf as fitz
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_scan  # noqa: E402
from bookpdf import cli, ocr_tesseract, redo  # noqa: E402
from bookpdf.ocr import QuotaExhausted  # noqa: E402

needs_tess = pytest.mark.skipif(ocr_tesseract.check_available("kor+eng") is not None, reason="tesseract(kor) 없음")


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr("bookpdf.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_RPD", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


def _work(tmp_path, confs: dict[int, float], engine="tesseract", **extra):
    """work/pages 에 쪽별 결과를 만들어 (results, work) 를 돌려준다."""
    work = tmp_path / "work"
    (work / "pages").mkdir(parents=True)
    results = []
    for p, c in confs.items():
        img = tmp_path / f"img{p}.png"
        Image.new("RGB", (400, 600), "white").save(img)
        r = {"page": p, "markdown": f"원본{p}", "figures": [], "confidence": c, "engine": engine, "image": str(img), **extra}
        (work / "pages" / f"p{p:04d}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
        (work / "pages" / f"p{p:04d}.md").write_text(r["markdown"], encoding="utf-8")
        results.append(r)
    return results, work


def _fake(conf=0.95, calls=None, text="새결과{}"):
    def f(img: Path) -> dict:
        n = int(img.stem.replace("img", ""))
        if calls is not None:
            calls.append(n)
        return {"markdown": text.format(n), "figures": [], "confidence": conf, "page_number": None, "engine": "gemini"}
    return f


# ---------- 단위 ----------
def test_only_pages_below_threshold_are_redone_and_old_results_are_preserved(tmp_path):
    results, work = _work(tmp_path, {1: 0.9, 2: 0.5, 3: 0.69, 4: 0.7})
    calls, logs = [], []
    out = redo.redo_low_pages(results, work, _fake(calls=calls), 0.7, "gemini-x", log=logs.append)
    assert calls == [2, 3]                                     # 0.7 은 '미만'이 아님
    assert out["replaced"] == 2 and out["kept"] == 0
    assert [r["engine"] for r in results] == ["tesseract", "gemini", "gemini", "tesseract"]
    assert results[1]["markdown"] == "새결과2" and results[1]["model"] == "gemini-x"
    assert results[1]["redo"]["from_engine"] == "tesseract" and results[1]["redo"]["from_conf"] == 0.5
    saved = json.loads((work / "pages" / "p0002.json").read_text(encoding="utf-8"))
    assert saved["engine"] == "gemini"                         # 디스크도 교체(다음 실행에서 캐시로 사용)
    old = json.loads((work / "pages_prev" / "p0002.v1.json").read_text(encoding="utf-8"))
    assert old["engine"] == "tesseract" and old["markdown"] == "원본2"
    assert (work / "pages_prev" / "p0002.v1.md").read_text(encoding="utf-8") == "원본2"
    assert not (work / "pages_prev" / "p0001.v1.json").exists()
    assert any("교체" in l and "p2" in l and "v1" in l for l in logs)


def test_worse_or_empty_or_failed_new_result_keeps_old(tmp_path):
    results, work = _work(tmp_path, {1: 0.5, 2: 0.5, 3: 0.5})
    logs = []

    def ocr(img):
        n = int(img.stem[3:])
        if n == 1:
            return {"markdown": "더 낮음", "figures": [], "confidence": 0.3, "engine": "gemini"}
        if n == 2:
            return {"markdown": "  ", "figures": [], "confidence": 0.99, "engine": "gemini"}
        raise RuntimeError("503 high demand")

    out = redo.redo_low_pages(results, work, ocr, 0.7, "m", log=logs.append)
    assert out["kept"] == 2 and out["failed"] == 1 and out["replaced"] == 0
    assert all(r["engine"] == "tesseract" and r["markdown"].startswith("원본") for r in results)
    assert json.loads((work / "pages" / "p0001.json").read_text(encoding="utf-8"))["markdown"] == "원본1"
    assert not (work / "pages_prev").exists()                  # 교체가 없으면 백업도 만들지 않음
    st = redo.load_state(work)
    assert set(st) == {"1", "2"} and not st["1"]["replaced"]   # 실패한 쪽(3)은 기록하지 않아 다음 실행에서 재시도
    assert any("기존 유지" in l and "p1" in l for l in logs) and any("실패" in l and "p3" in l for l in logs)


def test_second_run_skips_pages_already_tried_with_same_model_but_not_with_other_model(tmp_path):
    results, work = _work(tmp_path, {1: 0.5, 2: 0.5})
    calls = []
    redo.redo_low_pages(results, work, _fake(0.1, calls), 0.7, "m1", log=lambda *_: None)     # 둘 다 유지
    assert calls == [1, 2]
    calls.clear()
    redo.redo_low_pages(results, work, _fake(0.1, calls), 0.7, "m1", log=lambda *_: None)
    assert calls == []                                          # 같은 모델: 호출 없음
    redo.redo_low_pages(results, work, _fake(0.9, calls), 0.7, "m2", log=lambda *_: None)
    assert calls == [1, 2]                                      # 다른 모델: 다시 시도
    assert (work / "pages_prev" / "p0001.v1.json").exists()


def test_backup_versions_accumulate(tmp_path):
    results, work = _work(tmp_path, {1: 0.2})
    redo.redo_low_pages(results, work, _fake(0.5), 0.9, "m1", log=lambda *_: None)
    redo.redo_low_pages(results, work, _fake(0.8, text="셋째{}"), 0.9, "m2", log=lambda *_: None)
    v1 = json.loads((work / "pages_prev" / "p0001.v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((work / "pages_prev" / "p0001.v2.json").read_text(encoding="utf-8"))
    assert v1["engine"] == "tesseract" and v2["engine"] == "gemini" and v2["model"] == "m1"
    assert results[0]["markdown"] == "셋째1"


def test_quota_stop_keeps_done_work_and_resumes(tmp_path):
    results, work = _work(tmp_path, {1: 0.2, 2: 0.2, 3: 0.2})
    n = {"c": 0}

    def ocr(img):
        n["c"] += 1
        if n["c"] == 2:
            raise QuotaExhausted("일일 한도(RPD) 도달: x", "daily")
        return _fake()(img)

    logs = []
    out = redo.redo_low_pages(results, work, ocr, 0.7, "m", log=logs.append)
    assert out["replaced"] == 1 and out["stopped"] and "일일 한도" in out["stopped"]
    assert [r["engine"] for r in results] == ["gemini", "tesseract", "tesseract"]
    calls = []
    redo.redo_low_pages(results, work, _fake(calls=calls), 0.7, "m", log=lambda *_: None)
    assert calls == [2, 3]                                      # 이어서: 처리된 쪽 제외


def test_max_calls_limit(tmp_path):
    results, work = _work(tmp_path, {1: 0.2, 2: 0.2, 3: 0.2})
    calls = []
    out = redo.redo_low_pages(results, work, _fake(calls=calls), 0.7, "m", max_calls=2, log=lambda *_: None)
    assert calls == [1, 2] and "호출 상한" in out["stopped"]


def test_skips_picture_pages_and_error_pages(tmp_path):
    results, work = _work(tmp_path, {1: 0.2, 2: 0.2, 3: 0.2})
    results[1]["chars"] = 0                                     # 그림 쪽 표식
    results[2]["error"] = "x"
    calls, logs = [], []
    redo.redo_low_pages(results, work, _fake(calls=calls), 0.7, "m", log=logs.append,
                        skip=lambda r: r.get("chars") == 0)
    assert calls == [1]
    assert any("그림 쪽" in l and "p2" in l for l in logs) and any("처리 실패" in l and "p3" in l for l in logs)


def test_new_figures_do_not_overwrite_old_ones(tmp_path):
    results, work = _work(tmp_path, {1: 0.2})
    (work / "figures").mkdir()
    (work / "figures" / "p0001_1.png").write_bytes(b"old")

    def ocr(img):
        return {"markdown": "표 있음", "confidence": 0.9, "engine": "gemini",
                "figures": [{"box_2d": [100, 100, 600, 600], "caption": "c"}]}

    redo.redo_low_pages(results, work, ocr, 0.7, "m", log=lambda *_: None)
    assert (work / "figures" / "p0001_1.png").read_bytes() == b"old"
    assert (work / "figures_redo" / "p0001_1.png").exists() and results[0]["figures"][0]["path"].endswith("figures_redo/p0001_1.png")


def test_report_lines(tmp_path):
    results, work = _work(tmp_path, {1: 0.5, 2: 0.5})
    redo.redo_low_pages(results, work, lambda img: _fake(0.9 if "1" in img.name else 0.1)(img), 0.7, "m", log=lambda *_: None)
    txt = "\n".join(redo.report_lines(work))
    assert "| p1 | m | tesseract 0.5 | 0.9 | 교체 |" in txt and "| p2 | m | tesseract 0.5 | 0.1 | 유지 |" in txt
    assert "같은 척도가 아닙니다" in txt
    assert redo.report_lines(tmp_path / "none") == []


# ---------- CLI ----------
def test_cli_rejects_bad_values_and_missing_key(tmp_path, monkeypatch, capsys):
    pdf = tmp_path / "x.pdf"
    d = fitz.open()
    d.new_page()
    d.save(pdf)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert cli.main([str(pdf), "--out", str(tmp_path / "o"), "--redo-below", "0"]) == 2
    assert cli.main([str(pdf), "--out", str(tmp_path / "o"), "--redo-below", "1.5"]) == 2
    assert cli.main([str(pdf), "--out", str(tmp_path / "o"), "--redo-below", "0.7"]) == 2      # 키 없음
    err = capsys.readouterr().err
    assert "GEMINI_API_KEY" in err and "0 초과 1 이하" in err
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert cli.main([str(pdf), "--out", str(tmp_path / "o"), "--redo-below", "0.7", "--split-only"]) == 2


@pytest.fixture(scope="module")
def scan_pdf(tmp_path_factory):
    d = tmp_path_factory.mktemp("scan")
    make_scan.build_scan_pdf(d / "scan.pdf")
    return d / "scan.pdf"


@needs_tess
def test_cli_redo_replaces_low_pages_in_tesseract_book(scan_pdf, tmp_path, monkeypatch, capsys):
    out = tmp_path / "out"
    base = [str(scan_pdf), "--ocr", "tesseract", "--dpi", "300", "--embed-page-images", "--out", str(out), "--name", "b.pdf"]
    assert cli.main(base) in (0, 4)
    capsys.readouterr()
    before = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (out / "work" / "pages").glob("p*.json")}
    assert all(r["engine"] == "tesseract" for r in before.values())

    seen, made = [], {}

    def maker(key, model, rpm, *a, **k):
        made["model"] = model

        def f(img):
            seen.append(Path(img).name)
            return {"markdown": f"## 재처리 제목\n\n재처리본 {Path(img).stem} 가나다라마바사아", "figures": [],
                    "confidence": 0.99, "page_number": None, "engine": "gemini", "model": model}
        return f

    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", maker)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    rc = cli.main(base + ["--redo-below", "0.99"])
    cap = capsys.readouterr().out
    assert rc in (0, 4), cap
    assert made["model"] == "gemini-test-model"                  # GEMINI_MODEL 사용
    assert seen and "p0003.png" not in seen                       # 3쪽(차트)은 원본 이미지로 삽입되는 쪽이라 제외
    assert len(seen) < len(before)
    after = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (out / "work" / "pages").glob("p*.json")}
    redone = {n for n, r in after.items() if r["engine"] == "gemini"}
    assert redone == {n.replace(".png", ".json") for n in seen}
    for n in redone:                                               # 기존 결과 보존
        v1 = json.loads((out / "work" / "pages_prev" / n.replace(".json", ".v1.json")).read_text(encoding="utf-8"))
        assert v1 == before[n]
    assert "재처리 대상" in cap and "교체" in cap
    body = (out / "전체원문.md").read_text(encoding="utf-8")
    assert "재처리본 p00" in body
    assert "## 재처리 기록 (--redo-below)" in (out / "리포트.md").read_text(encoding="utf-8")
    text = "".join(p.get_text() for p in fitz.open(out / "b.pdf"))
    assert "재처리본" in text

    n_calls = len(seen)                                            # 같은 명령 재실행: 다시 호출하지 않음
    assert cli.main(base + ["--redo-below", "0.99"]) in (0, 4)
    assert len(seen) == n_calls
    assert cli.main(base) in (0, 4)                                # 옵션 없이 재조립해도 교체 결과 유지
    assert "재처리본 p00" in (out / "전체원문.md").read_text(encoding="utf-8")
