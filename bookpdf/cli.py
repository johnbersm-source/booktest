"""명령줄: python -m bookpdf INPUT.pdf [--out output] [--title 제목] ..."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import pdf_input, pipeline, postprocess, render, report
from .config import load_settings


def build_figure_map(results: list[dict]) -> dict[str, dict]:
    """페이지별 그림을 전역 고유키 'p{page}_{n}' 로 매핑."""
    fm = {}
    for r in results:
        for f in r.get("figures", []):
            box = f["box_2d"]
            fm[f"p{r['page']}_{f['n']}"] = {
                "path": f["path"], "caption": f.get("caption", ""),
                "frac_w": (box[3] - box[1]) / 1000,
            }
    return fm


def namespaced_markdown(r: dict) -> str:
    return re.sub(r"\{\{FIG:(\d+)\}\}", lambda m: "{{FIG:p%d_%s}}" % (r["page"], m.group(1)), r.get("markdown", ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bookpdf", description="PDF(또는 로컬 전용 .mov)에서 글자를 추출해 책 PDF 생성")
    ap.add_argument("input", type=Path, help="입력 .pdf (로컬 전용: .mov)")
    ap.add_argument("--out", type=Path, default=Path("output"), help="출력 폴더 (기본: output)")
    ap.add_argument("--work", type=Path, default=None, help="중간 결과 폴더 (기본: <out>/work). 같은 폴더로 재실행하면 이어서 진행")
    ap.add_argument("--name", default="책.pdf", help="출력 PDF 파일명 (기본: 책.pdf)")
    ap.add_argument("--title", default=None, help="표지 제목 (기본: 입력 파일명)")
    ap.add_argument("--dpi", type=int, default=200, help="PDF 페이지 렌더 DPI")
    ap.add_argument("--skip-ocr-if-text", action="store_true", help="텍스트 레이어가 있는 PDF 페이지는 OCR 생략")
    ap.add_argument("--threshold", type=float, default=0.7, help="신뢰도 낮음 기준 (기본 0.7)")
    ap.add_argument("--fps", type=float, default=3.0, help="[.mov 로컬 전용] 초당 추출 프레임 수 (2~5 권장)")
    ap.add_argument("--page-thr", type=float, default=0.10,
                    help="[.mov 로컬 전용] 페이지 구분 민감도(0~1). 같은 페이지가 쪼개지면 올리고, 다른 페이지가 합쳐지면 내리세요")
    ap.add_argument("--max-pages", type=int, default=0, help="앞쪽 N페이지만 처리(테스트용)")
    a = ap.parse_args(argv)

    inp: Path = a.input
    out: Path = a.out
    work: Path = a.work or (out / "work")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    s = load_settings()
    extra: list[str] = []
    pdf_for_text: Path | None = None

    # ---- 1. 입력 검증 + 페이지 이미지화
    ext = inp.suffix.lower()
    if ext == ".pdf":
        try:
            n = pdf_input.validate_pdf(inp)
        except ValueError as e:
            print(f"[입력 오류] {e}", file=sys.stderr)
            return 2
        print(f"입력 PDF 확인: {n}쪽")
        images = pdf_input.render_pages(inp, work / "page_images", a.dpi)
        pdf_for_text = inp
    elif ext in (".mov", ".mp4", ".m4v"):
        try:
            from .local import video_input  # 로컬 전용(opencv/ffmpeg 필요)
        except ImportError as e:
            print(f"[영상 입력은 로컬 전용] 의존성이 없습니다: {e}\n  pip install -r requirements-video.txt 와 ffmpeg 설치가 필요합니다.", file=sys.stderr)
            return 2
        try:
            images, vinfo = video_input.extract_pages(inp, work / "page_images", fps=a.fps, same_thr=a.page_thr)
        except (RuntimeError, ValueError) as e:
            print(f"[영상 처리 오류] {e}", file=sys.stderr)
            return 2
        extra += ["## 영상 입력 정보", ""] + [f"- {k}: {v}" for k, v in vinfo.items()]
    else:
        print(f"[입력 오류] 지원하지 않는 형식입니다: {ext} (.pdf, 로컬 전용 .mov)", file=sys.stderr)
        return 2
    if a.max_pages:
        images = images[: a.max_pages]

    # ---- 2. OCR (Gemini) / 텍스트 레이어
    ocr = None
    if s.api_key:
        from .ocr import make_gemini_ocr
        ocr = make_gemini_ocr(s.api_key, s.model, s.rpm)
        print(f"Gemini 사용: 모델 {s.model}, RPM {s.rpm}")
    else:
        print("GEMINI_API_KEY 가 없습니다(.env 확인). 텍스트 레이어가 있는 PDF 페이지만 처리됩니다.")
    results = pipeline.process_pages(images, work, ocr, pdf=pdf_for_text, use_textlayer=a.skip_ocr_if_text)

    # ---- 3. 후처리
    body, stats = postprocess.postprocess([namespaced_markdown(r) for r in results])
    (out / "전체원문.md").write_text(body, encoding="utf-8")
    md_dir = out / "pages_md"
    md_dir.mkdir(exist_ok=True)
    for r in results:
        (md_dir / f"p{r['page']:04d}.md").write_text(r.get("markdown", ""), encoding="utf-8")

    # ---- 4. 책 PDF
    low = report.low_confidence_pages(results, a.threshold)
    if not body.strip():
        print("[결과 오류] 추출된 텍스트가 없어 책 PDF를 만들지 않습니다. (API 키/입력을 확인하세요)", file=sys.stderr)
        report.write_report(out / "리포트.md", results, low, stats, extra)
        return 3
    title = a.title or inp.stem
    html_str = render.build_html(body, build_figure_map(results), title, render.default_subtitle())
    pdf_path = out / a.name
    render.build_pdf(html_str, pdf_path, allowed_dirs=[work])

    # ---- 5. 결과 검증 + 리포트
    report.write_report(out / "리포트.md", results, low, stats, extra)
    problems = _verify(pdf_path, results)
    print(f"완료: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
    print(f"신뢰도 낮은 페이지 {len(low)}개 → {out / '리포트.md'}")
    for p in problems:
        print(f"[검증 경고] {p}")
    return 0 if not problems else 4


def _verify(pdf: Path, results: list[dict]) -> list[str]:
    """생성된 PDF 를 다시 열어 페이지 수/글자/그림/폰트 포함 여부를 확인."""
    import pymupdf as fitz
    issues = []
    doc = fitz.open(pdf)
    if doc.page_count < 2:
        issues.append("PDF 페이지 수가 너무 적습니다.")
    text = "".join(p.get_text() for p in doc)
    imgs = sum(len(p.get_images()) for p in doc)
    expect_imgs = sum(len(r.get("figures", [])) for r in results)
    if not text.strip():
        issues.append("PDF 에서 글자를 읽을 수 없습니다(폰트 문제 가능).")
    if expect_imgs and imgs == 0:
        issues.append(f"그림 {expect_imgs}개가 있어야 하지만 PDF 에 이미지가 없습니다.")
    fonts = {f[3] for p in doc for f in p.get_fonts()}
    if not any("Nanum" in f for f in fonts):
        issues.append(f"NanumGothic 폰트가 포함되지 않았습니다(포함된 폰트: {sorted(fonts)}).")
    doc.close()
    return issues


if __name__ == "__main__":
    raise SystemExit(main())
