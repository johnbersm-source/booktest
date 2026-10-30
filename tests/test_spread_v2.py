"""펼침면 판정 개선: 점수/임계값, 세로 프레임, 책 전체 제본선 prior, 판정 로그, --split-only, 쪽번호 원인 구분."""
import csv
import json
import sys
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import make_spread as ms  # noqa: E402
from bookpdf import cli, report, spread  # noqa: E402


def frames_pdf(path: Path, imgs: list[Image.Image]) -> Path:
    """이미지를 쪽 크기 그대로(200dpi 렌더 시 원래 픽셀) 한 장씩 넣은 PDF. 영상 프레임을 PDF 로 묶은 입력을 흉내낸다."""
    doc = pymupdf.open()
    for im in imgs:
        png = path.with_suffix(".tmp.png")
        im.save(png)
        pg = doc.new_page(width=im.width * 0.36, height=im.height * 0.36)
        pg.insert_image(pg.rect, filename=str(png))
    doc.save(path)
    png.unlink()
    return path


def portrait_single(seed: int) -> Image.Image:
    return ms.to_portrait_frame(ms.page_img(seed)[0].resize((560, 800)))


# ---- 점수/판정 ----------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_portrait_frame_spreads_are_split_but_single_pages_are_not(seed):
    for im in (ms.to_portrait_frame(ms.make_spread(seed=seed)[0]),
               ms.to_portrait_frame(ms.make_spread(seed=seed, tilt_deg=2.0)[0]),
               ms.to_portrait_frame(ms.make_spread(seed=seed)[0], cx=0.46)):
        a = spread.analyze(im)
        assert a.aspect < 0.7 and a.score >= 0.5 and a.gutter.shadow > 0.1 and a.gutter.gap > 0.05, a
        assert 0.40 < a.gutter.x < 0.60
    assert spread.analyze(portrait_single(seed)).score < 0.2
    assert spread.analyze(ms.page_img(seed)[0]).score < 0.2


def test_old_rule_missed_portrait_frames_new_rule_does_not():
    """대조: 이전 판정(가로세로비 ≥ 1.25 필수)은 세로 프레임 펼침을 전부 놓친다."""
    im = ms.to_portrait_frame(ms.make_spread(seed=1)[0])
    assert im.width / im.height < spread.MIN_ASPECT      # 이전 규칙이면 여기서 곧바로 '단일'
    assert spread.is_spread(im, "auto")[0] is True


def test_two_column_portrait_page_is_not_a_spread():
    """두 단으로 짠 일반 쪽: 가운데 글자 없는 띠는 있지만 그림자가 없으니 세로 이미지에서는 분리하지 않는다."""
    from bookpdf.config import FONT_REGULAR
    from PIL import ImageFont
    im = Image.new("RGB", (800, 1100), (250, 248, 240))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(str(FONT_REGULAR), 24)
    for y in range(80, 1000, 38):
        d.text((60, y), "바다 하늘 구름 나무 사랑", fill=(20, 20, 20), font=f)
        d.text((470, y), "시간 마음 기억 도시 여행", fill=(20, 20, 20), font=f)
    a = spread.analyze(im)
    assert a.gutter.gap > 0.03 and a.score < 0.2, a        # 띠는 보이지만 점수는 낮다


def test_dark_chart_content_is_not_mistaken_for_a_shadow():
    """대조: 큰 어두운 그림이 한쪽에 있어도 '제본 그림자(양옆보다 움푹한 골짜기)'로 세지 않는다."""
    im = portrait_single(7)
    d = ImageDraw.Draw(im)
    d.rectangle([430, 560, 1000, 1400], fill=(25, 40, 90))      # 중앙부터 오른쪽까지 어두운 면
    assert spread.analyze(im).score < 0.5


def test_spread_thr_moves_the_decision():
    im = ms.to_portrait_frame(ms.make_chart_spread(seed=1))       # 점수 ≈ 0.68
    a = spread.analyze(im)
    assert 0.6 < a.score < 0.75
    assert spread.is_spread(im, "auto", 0.5)[0] is True
    assert spread.is_spread(im, "auto", 0.8)[0] is False
    assert spread.is_spread(im, "auto", 0.3)[0] is True


def test_prior_rescues_weak_page_but_not_a_cover():
    """책 전체에서 확실한 프레임들의 제본선 위치에 약한 그림자가 있으면 분리하고, 가운데에 글자가 있는 단일 쪽은 그대로 둔다."""
    strong = [ms.to_portrait_frame(ms.make_spread(seed=s)[0]) for s in (1, 2, 3, 4)]
    weak_chart = ms.to_portrait_frame(ms.make_chart_spread(seed=5, shadow_amp=0.05))
    cover = portrait_single(9)
    items = [spread.analyze(im) for im in strong + [weak_chart, cover]]
    assert items[4].score < 0.5, "전제: 약한 차트 쪽은 자체 점수로는 임계 미만"
    stat = spread.decide(items, "auto", 0.5)
    assert stat["prior_n"] == 4 and stat["promoted"] == 1
    assert items[4].decision and items[4].prior and "제본선 위치" in items[4].reason
    assert items[4].gutter.x == pytest.approx(items[0].gutter.x, abs=0.03)
    assert not items[5].decision and not items[5].prior


def test_prior_needs_enough_strong_frames():
    """확실한 프레임이 3장 미만이면 prior 를 쓰지 않는다(우연한 한두 장으로 책 전체를 단정하지 않음)."""
    items = [spread.analyze(im) for im in
             [ms.to_portrait_frame(ms.make_spread(seed=1)[0]), ms.to_portrait_frame(ms.make_chart_spread(seed=5, shadow_amp=0.05)),
              portrait_single(3), portrait_single(4)]]
    stat = spread.decide(items, "auto", 0.5)
    assert stat["prior_x"] is None and stat["promoted"] == 0 and not items[1].decision


def test_off_and_on_modes_ignore_score():
    items = [spread.analyze(portrait_single(2))]
    spread.decide(items, "on"); assert items[0].decision
    spread.decide(items, "off"); assert not items[0].decision


# ---- 판정 로그 -----------------------------------------------------------------------------

def test_frame_log_csv_has_one_row_per_frame(tmp_path):
    srcs = []
    for i, im in enumerate([ms.to_portrait_frame(ms.make_spread(seed=1)[0]), portrait_single(2),
                            ms.make_spread(seed=3)[0]], start=1):
        p = tmp_path / f"p{i:04d}.png"; im.save(p); srcs.append(p)
    logs: list[str] = []
    pages, infos, frames = spread.expand_spreads_ex(srcs, tmp_path / "pi", "auto", log=logs.append, verbose=True)
    assert [f["decision"] for f in frames] == ["split", "single", "split"]
    rows = list(csv.DictReader(open(tmp_path / "pi" / "spread_log.csv", encoding="utf-8-sig")))
    assert len(rows) == 3 and rows[0]["source"] == "p0001.png" and rows[0]["decision"] == "split"
    assert float(rows[0]["score"]) >= 0.5 and rows[1]["reason"].startswith("점수") and "제본 증거 없음" in rows[1]["reason"]
    assert len([l for l in logs if "프레임" in l]) == 3
    assert len(pages) == 5 and [i["kind"] for i in infos] == ["left", "right", "single", "left", "right"]
    assert all("evidence" in i and "aspect" in i for i in infos)


# ---- --split-only ----------------------------------------------------------------------------

def _forbid_ocr(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("--split-only 에서 OCR 엔진을 만들면 안 됨")
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", boom)
    monkeypatch.setattr("bookpdf.ocr_tesseract.make_tesseract_ocr", boom)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")


def test_split_only_end_to_end(tmp_path, monkeypatch):
    _forbid_ocr(monkeypatch)
    imgs = [ms.to_portrait_frame(ms.make_spread(seed=s)[0]) for s in (1, 2, 3)] + [portrait_single(4)]
    pdf = frames_pdf(tmp_path / "frames.pdf", imgs)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    assert cli.main([str(pdf), "--out", str(out), "--split-only", "--ocr", "tesseract"]) == 0     # tesseract 미설치여도 통과해야 함
    sp = out / "work" / "split_only"
    pages = sorted((sp / "page_images").glob("p*.png"))
    assert len(pages) == 7                                               # 펼침면 3장 → 6쪽 + 단일 1쪽
    assert (sp / "page_images" / "spread_log.csv").exists() and list((sp / "preview").glob("preview_*.jpg"))
    assert not (out / "work" / "pages").exists() and not (out / "책.pdf").exists(), "OCR/책 PDF 산출물이 생기면 안 됨"
    md = (out / "분리점검.md").read_text(encoding="utf-8")
    assert "원본 4장 → 책 7쪽" in md and "4장 중 3장 분리" in md and "--spread-thr" in md
    # 좌/우 쪽이 실제로 잘렸는지: 왼쪽 쪽은 빨강 표식만, 오른쪽 쪽은 파랑 표식만 많다
    left, right = Image.open(pages[0]), Image.open(pages[1])
    assert ms.count_color(left, "red") > 3 * ms.count_color(left, "blue")
    assert ms.count_color(right, "blue") > 3 * ms.count_color(right, "red")
    # 임계값만 바꿔 다시 실행: 설정 불일치로 막히지 않고, 결과 폴더는 새로 만들어진다 (약한 점수 ≈0.68 의 차트 프레임으로 확인)
    cpdf = frames_pdf(tmp_path / "charts.pdf", [ms.to_portrait_frame(ms.make_chart_spread(seed=s)) for s in (1, 2)])
    out2 = tmp_path / "out2"; sp2 = out2 / "work" / "split_only" / "page_images"
    assert cli.main([str(cpdf), "--out", str(out2), "--split-only"]) == 0
    assert len(list(sp2.glob("p*.png"))) == 4
    assert cli.main([str(cpdf), "--out", str(out2), "--split-only", "--spread-thr", "0.8"]) == 0
    assert len(list(sp2.glob("p*.png"))) == 2
    assert "2장 중 0장 분리" in (out2 / "분리점검.md").read_text(encoding="utf-8")
    assert cli.main([str(cpdf), "--out", str(out2), "--split-only", "--spread-thr", "1.5"]) == 2


def test_split_only_then_normal_run_with_same_work(tmp_path, monkeypatch):
    """--split-only 로 시험한 뒤 같은 work 에서 본 실행이 설정 불일치 없이 이어지고, 같은 판정으로 분리한다."""
    imgs = [ms.to_portrait_frame(ms.make_spread(seed=s)[0]) for s in (1, 2)] + [portrait_single(4)]
    pdf = frames_pdf(tmp_path / "frames.pdf", imgs)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    assert cli.main([str(pdf), "--out", str(out), "--split-only", "--spread-thr", "0.4"]) == 0
    calls: list = []

    def fake(img):
        calls.append(img.name)
        return {"markdown": f"## 쪽{len(calls)}\n\n본문 {len(calls)} 가나다라마바사아자차", "figures": [], "confidence": 0.9,
                "page_number": len(calls), "engine": "gemini"}
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: fake)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    assert cli.main([str(pdf), "--out", str(out), "--spread-thr", "0.4"]) == 0
    assert len(calls) == 5                                                # 2*2 + 1
    rep = (out / "리포트.md").read_text(encoding="utf-8")
    assert "펼침면 판정 요약" in rep and "`--spread-thr 0.4`" in rep and "3장 중 2장 분리" in rep
    meta = json.loads((out / "work" / "meta.json").read_text(encoding="utf-8"))
    assert "_partial" not in meta and meta["spread_thr"] == 0.4
    # 이제 임계값을 바꾸면 본 실행은 이전 결과와 섞이지 않도록 막힌다
    assert cli.main([str(pdf), "--out", str(out), "--spread-thr", "0.6"]) == 2


def test_meta_from_older_version_requires_restart(tmp_path, monkeypatch, capsys):
    pdf = frames_pdf(tmp_path / "frames.pdf", [portrait_single(4), portrait_single(5)])
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"; (out / "work").mkdir(parents=True)
    st = pdf.stat()
    old = {"input": pdf.name, "size": st.st_size, "mtime": st.st_mtime_ns, "spread": "auto", "gutter_offset": 0.0,
           "spread_overlap": 0.015, "ocr": "gemini", "pages": None, "dpi": 200, "fps": 3.0, "page_thr": 0.1, "max_frame_side": 2000}
    (out / "work" / "meta.json").write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    assert cli.main([str(pdf), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "spread_algo" in err and "--restart" in err


# ---- 쪽번호: 분리 실패 vs 촬영 누락 ---------------------------------------------------------

def res(n_by_page: dict[int, int | None]) -> list[dict]:
    return [{"page": p, "page_number": n, "markdown": "x", "confidence": 0.9} for p, n in n_by_page.items()]


def info(page: int, kind: str, evidence: float = 0.0, aspect: float = 0.7) -> dict:
    return {"page": page, "source": page, "kind": kind, "evidence": evidence, "aspect": aspect}


def test_page_number_skips_are_attributed_to_split_failure_or_capture_loss():
    """사용자가 본 256→258→260→263: 앞의 두 번 건너뜀은 펼침면 미분리, 마지막 3 건너뜀은 미분리 1 + 촬영 누락 1."""
    results = res({1: 254, 2: 256, 3: 258, 4: 260, 5: 263, 6: 264})
    spreads = [info(1, "single", 0.9), info(2, "single", 0.9), info(3, "single", 0.9), info(4, "single", 0.9),
               info(5, "single", 0.0, 0.56), info(6, "single", 0.0, 0.56)]
    chk = report.check_page_numbers(results, spreads)
    m = {x["between_pages"]: x for x in chk["missing"]}
    assert m[(1, 2)]["split_suspect"] == 1 and m[(1, 2)]["capture_missing"] == 0 and m[(1, 2)]["split_pages"] == [1]
    assert m[(3, 4)]["split_suspect"] == 1 and m[(3, 4)]["capture_missing"] == 0
    assert m[(4, 5)]["count"] == 2 and m[(4, 5)]["split_suspect"] == 1 and m[(4, 5)]["capture_missing"] == 1
    txt = "\n".join(report.format_page_number_report(chk))
    assert "분리 실패 의심" in txt and "촬영 누락 의심" in txt and "--spread-thr" in txt and "추정" in txt


def test_page_number_skip_without_split_evidence_is_capture_loss():
    """모두 정상 분리(left/right)된 구간에서 번호가 건너뛰면 분리 문제가 아니라 촬영 누락으로 본다."""
    results = res({1: 10, 2: 11, 3: 12, 4: 15, 5: 16})
    spreads = [info(1, "left"), info(2, "right"), info(3, "left"), info(4, "right"), info(5, "single", 0.0, 0.56)]
    chk = report.check_page_numbers(results, spreads)
    (m,) = chk["missing"]
    assert m["split_suspect"] == 0 and m["capture_missing"] == 2 and m["missing_numbers"] == [13, 14]
    txt = "\n".join(report.format_page_number_report(chk))
    assert "촬영 누락 의심" in txt and "분리 실패 의심" not in txt


def test_page_number_check_without_spread_info_keeps_old_wording():
    chk = report.check_page_numbers(res({1: 1, 2: 2, 3: 3, 4: 6}))
    txt = "\n".join(report.format_page_number_report(chk))
    assert "**누락 의심**" in txt and "분리 실패" not in txt


def test_full_run_report_classifies_unsplit_spreads(tmp_path, monkeypatch):
    """--spread off 로 일부러 분리를 끈 실행: 번호가 2씩 건너뛰는 것이 '분리 실패 의심' 으로 리포트에 나온다."""
    imgs = [ms.make_spread(seed=s)[0] for s in (1, 2, 3, 4)]               # 가로 펼침 4장, 분리하지 않음
    pdf = frames_pdf(tmp_path / "frames.pdf", imgs)
    nums = iter([10, 12, 14, 16])

    def fake(img):
        n = next(nums)
        return {"markdown": f"## 제{n}장\n\n본문 {n}쪽 가나다라마바사아자차 {'카타파하' * (n % 5 + 2)}", "figures": [], "confidence": 0.9,
                "page_number": n, "engine": "gemini"}
    monkeypatch.setattr("bookpdf.ocr.make_gemini_ocr", lambda *a, **k: fake)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy")
    monkeypatch.chdir(tmp_path)
    # --spread off 면 판정을 하지 않아 증거 값이 없다 → 가로 비율(≥1.10)만으로 의심. 판정 로그에는 'off' 가 남는다.
    assert cli.main([str(pdf), "--out", str(tmp_path / "o"), "--spread", "off"]) == 0
    rep = (tmp_path / "o" / "리포트.md").read_text(encoding="utf-8")
    assert "분리 실패 의심" in rep and "촬영 누락 의심" not in rep
