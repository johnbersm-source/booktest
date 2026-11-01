"""페이지별 처리(OCR/텍스트레이어 → 그림 크롭 → 디스크 저장). 중단 후 이어서 실행 가능."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import pymupdf as fitz
from PIL import Image

from . import textlayer
from .ocr import KIND_LABEL, QuotaExhausted


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


def write_quota_hint(work: Path, kind: str, ok_calls: int, message: str) -> None:
    """한도에 걸린 기록. 다음 실행에서 --max-calls 를 얼마로 둘지 안내하는 데 쓴다."""
    (work / "quota_hint.json").write_text(json.dumps(
        {"kind": kind, "successful_calls": max(0, ok_calls), "at": time.strftime("%Y-%m-%d %H:%M:%S"),
         "message": message[:300]}, ensure_ascii=False, indent=1), encoding="utf-8")


def read_quota_hint(work: Path) -> dict | None:
    p = work / "quota_hint.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


def fmt_duration(sec: float) -> str:
    sec = max(0, int(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}시간 {m}분"
    if m:
        return f"{m}분 {s}초"
    return f"{s}초"


def process_pages(
    page_images: list[Path],
    work: Path,
    ocr: Callable[[Path], dict] | None,
    pdf: Path | None = None,
    use_textlayer: bool = False,
    log: Callable[[str], None] = print,
    max_calls: int = 0,
    rpm: float = 0.0,
    clock: Callable[[], float] = time.monotonic,
    pdf_page_map: list[int] | None = None,
) -> tuple[list[dict], str | None]:
    """각 페이지 결과를 work/pages/pNNNN.json 으로 저장. 이미 있으면 건너뛴다.

    반환: (결과 목록, 중단 사유). 중단 사유는 일일 한도 도달/호출 상한 도달 시에만 문자열,
    정상이면 None. pdf_page_map[i-1] 은 i번째 쪽에 대응하는 PDF 쪽 번호(1부터, --pages 사용 시). 중단되면 남은 쪽은 결과에 포함되지 않으며, 같은 명령으로 다시 실행하면 이어서 처리한다.
    """
    res_dir = work / "pages"
    res_dir.mkdir(parents=True, exist_ok=True)
    progress_path = work / "progress.json"
    doc = fitz.open(pdf) if (pdf and use_textlayer) else None
    results: list[dict] = []
    total = len(page_images)
    cached = sum(1 for i in range(1, total + 1) if (res_dir / f"p{i:04d}.json").exists())
    todo = total - cached
    stop: str | None = None
    if todo and ocr is not None:
        floor = f", 호출 제한상 최소 약 {fmt_duration(todo * 60 / rpm)}" if rpm else ""
        log(f"처리 대상 {todo}쪽 (이미 완료 {cached}쪽){floor}")
    t0 = clock()
    ocr_calls = 0
    try:
        for i, img in enumerate(page_images, start=1):
            rp = res_dir / f"p{i:04d}.json"
            if rp.exists():
                results.append(json.loads(rp.read_text(encoding="utf-8")))
                continue
            res: dict | None = None
            if doc is not None:
                tl = textlayer.extract_page(doc[(pdf_page_map[i - 1] if pdf_page_map else i) - 1])
                if tl["markdown"].strip():
                    res = tl
            if res is None:
                if ocr is None:
                    # 캐시하지 않는다: 나중에 키를 넣고 재실행하면 이 페이지를 다시 처리해야 함
                    results.append({"page": i, "markdown": "", "figures": [], "confidence": 0.0,
                                    "engine": "none", "error": "OCR 엔진 없음(GEMINI_API_KEY 미설정, --ocr tesseract 로 대체 가능)"})
                    log(f"[{i}/{total}] OCR 엔진 없음, 건너뜀(캐시하지 않음)")
                    continue
                if max_calls and ocr_calls >= max_calls:
                    stop = f"이번 실행의 호출 상한({max_calls}회)에 도달"
                    break
                try:
                    ocr_calls += 1
                    res = ocr(img)
                except QuotaExhausted as e:
                    stop = str(e)
                    log(f"[{i}/{total}] [{KIND_LABEL.get(e.kind, e.kind)}] {e}")
                    write_quota_hint(work, e.kind, ocr_calls - 1, str(e))
                    break
                except Exception as e:  # noqa: BLE001  한 페이지 실패가 전체를 막지 않게
                    log(f"[{i}/{total}] OCR 실패: {e}")
                    # 실패는 저장하지 않아 다음 실행에서 재시도
                    results.append({"page": i, "markdown": "", "figures": [], "confidence": 0.0,
                                    "engine": "ocr", "error": str(e)})
                    continue
            res["page"] = i
            res["image"] = str(img)
            res["figures"] = crop_figures(img, res.get("figures", []), work / "figures", i)
            rp.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            (res_dir / f"p{i:04d}.md").write_text(res["markdown"], encoding="utf-8")
            done_now = sum(1 for _ in res_dir.glob("p*.json"))
            progress_path.write_text(json.dumps({"total": total, "done": done_now}), encoding="utf-8")
            results.append(res)
            elapsed = clock() - t0
            this_run = max(1, ocr_calls) if res["engine"] in ("gemini", "tesseract") else None
            msg = f"[{i}/{total}] {res['engine']} 완료 (신뢰도 {res['confidence']:.2f}, 그림 {len(res['figures'])})"
            if this_run and ocr_calls:
                per = elapsed / ocr_calls
                remain = total - done_now
                msg += f" | 쪽당 평균 {per:.1f}초, 남은 {remain}쪽 약 {fmt_duration(per * remain)}(예상)"
            log(msg)
    finally:
        if doc is not None:
            doc.close()
    if stop:
        left = total - sum(1 for _ in res_dir.glob("p*.json"))
        log(f"중단: {stop}. 남은 {left}쪽은 같은 명령을 다시 실행하면 이어서 처리됩니다.")
    return results, stop
