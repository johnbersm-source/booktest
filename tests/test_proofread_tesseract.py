"""실제 tesseract(kor+eng)가 있을 때만: tsv 단어 상자·신뢰도 경로를 합성 한글 이미지로 확인."""
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_proofread_fixtures as fx  # noqa: E402
from bookpdf import ocr_tesseract  # noqa: E402
from bookpdf.proofread import candidates as cd, detect as dt, normalize as nz, pipeline as pl, util  # noqa: E402

needs_tess = pytest.mark.skipif(ocr_tesseract.check_available("kor+eng") is not None, reason="tesseract(kor) 없음")

LINES = ["이동평균선은 일정 기간의 주가 평균을 이은 선이다.", "거래량이 늘면 추세의 신뢰도가 커진다.", "보조지표는 추세의 강도를 보여 준다."]


def draw(path: Path, lines, size=(1600, 400), px=44):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(str(fx.FONT), px)
    for i, ln in enumerate(lines):
        d.text((60, 40 + i * 100), ln, fill="black", font=f)
    im.save(path)
    return path


@needs_tess
def test_real_tesseract_tsv_boxes_and_pipeline(tmp_path):
    img = draw(tmp_path / "p.png", LINES)
    res = ocr_tesseract.make_tesseract_ocr()(img)
    work = tmp_path / "work"
    (work / "pages").mkdir(parents=True)
    rec = {**res, "page": 1, "image": str(img)}
    (work / "pages" / "p0001.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    page = util.load_pages(work)[0]
    cache = tmp_path / "tsv"
    words = dt.load_tsv(page, cache, run=True)
    assert words and (cache / "p0001.tsv").exists()                       # 실제 tesseract 로 tsv 생성·캐시
    assert all(0 <= w["box"][0] < w["box"][2] <= 1600 and 0 <= w["box"][1] < w["box"][3] <= 400 for w in words)
    lines = util.lines_of(nz.normalize_text(page["markdown"]).text)
    att = dt.attach_tsv(lines, words)
    ntok = sum(len(util.tokenize_line(l)) for l in lines)
    assert len(att) >= 0.6 * ntok, (len(att), ntok)                       # 본문 토큰 대부분이 tsv 단어와 짝지어짐
    # 두 번째 호출은 캐시를 읽고(tesseract 재실행 없이) 같은 결과
    assert dt.load_tsv(page, cache, run=False) == words
    # 파이프라인: tsv 사용 → 탐지 결과에 단어 상자(box)가 채워지고, 신뢰도 낮은 단어는 low_conf 로 잡힘
    opts = pl.Options(work=work, no_llm=True, no_verify=True, params=dt.Params(conf_thr=101.0))   # 모든 단어를 '낮음' 으로 간주해 경로 확인
    out = pl.run(opts, None, log=lambda *a: None)
    sp = out["detections"]["spans"]
    assert sp and all(s["box"] for s in sp if any(g["name"] == "low_conf" for g in s["signals"]))
    assert out["stats"]["mode"]["tsv_pages"] == 1
