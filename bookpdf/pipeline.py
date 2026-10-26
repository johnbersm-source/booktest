"""페이지별 처리(OCR/텍스트레이어 → 그림 크롭 → 디스크 저장). 중단 후 이어서 실행 가능."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pymupdf as fitz
from PIL import Image

from . import textlayer


def crop_figures(image_path: Path, figures: list[dict], fig_dir: Path, page_no: int) -> list[dict]:
    """box_2d(0~1000)로 원본 페이지 이미지에서 그림을 잘라 저장하고 경로를 채운다."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    out = []
    for n, f in enumerate(figures, start=1):
        ymin, xmin, ymax, xmax = f["box_2d"]
        box = (int(xmin / 1000 * W), int(ymin / 1000 * H), int(xmax / 1000 * W), int(ymax / 1000 * H))
        if box[2] - box[0] < 20 or box[3] - box[1] < 20:
            continue
        p = fig_dir / f"p{page_no:04d}_{n}.png"
        img.crop(box).save(p)
        out.append({**f, "n": n, "path": str(p)})
    return out


def process_pages(
    page_images: list[Path],
    work: Path,
    ocr: Callable[[Path], dict] | None,
    pdf: Path | None = None,
    use_textlayer: bool = False,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """각 페이지 결과를 work/pages/pNNNN.json 으로 저장. 이미 있으면 건너뛴다."""
    res_dir = work / "pages"
    res_dir.mkdir(parents=True, exist_ok=True)
    progress_path = work / "progress.json"
    doc = fitz.open(pdf) if (pdf and use_textlayer) else None
    results: list[dict] = []
    total = len(page_images)
    try:
        for i, img in enumerate(page_images, start=1):
            rp = res_dir / f"p{i:04d}.json"
            if rp.exists():
                results.append(json.loads(rp.read_text(encoding="utf-8")))
                log(f"[{i}/{total}] 이미 처리됨, 건너뜀")
                continue
            res: dict | None = None
            if doc is not None:
                tl = textlayer.extract_page(doc[i - 1])
                if tl["markdown"].strip():
                    res = tl
            if res is None:
                if ocr is None:
                    # 캐시하지 않는다: 나중에 키를 넣고 재실행하면 이 페이지를 다시 처리해야 함
                    results.append({"page": i, "markdown": "", "figures": [], "confidence": 0.0,
                                    "engine": "none", "error": "OCR 엔진 없음(GEMINI_API_KEY 미설정)"})
                    log(f"[{i}/{total}] OCR 엔진 없음, 건너뜀(캐시하지 않음)")
                    continue
                else:
                    try:
                        res = ocr(img)
                    except Exception as e:  # noqa: BLE001  한 페이지 실패가 전체를 막지 않게
                        log(f"[{i}/{total}] OCR 실패: {e}")
                        res = {"markdown": "", "figures": [], "confidence": 0.0, "engine": "gemini",
                               "error": str(e)}
                        # 실패는 저장하지 않아 다음 실행에서 재시도
                        res["page"] = i
                        results.append(res)
                        continue
            res["page"] = i
            res["image"] = str(img)
            res["figures"] = crop_figures(img, res.get("figures", []), work / "figures", i)
            rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            (res_dir / f"p{i:04d}.md").write_text(res["markdown"], encoding="utf-8")
            progress_path.write_text(
                json.dumps({"total": total, "done": sum(1 for _ in res_dir.glob("p*.json"))}), encoding="utf-8")
            log(f"[{i}/{total}] {res['engine']} 완료 (신뢰도 {res['confidence']:.2f}, 그림 {len(res['figures'])})")
            results.append(res)
    finally:
        if doc is not None:
            doc.close()
    return results
