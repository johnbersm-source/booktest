"""명령줄: python -m bookpdf INPUT.pdf [--out output] [--title 제목] ..."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from . import glyphs, pagesel, pdf_input, pipeline, postprocess, render, report, spread, spread_report, structure
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


SRC_KEYS = ("input", "size", "mtime", "pages", "dpi", "fps", "page_thr", "max_frame_side")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bookpdf", description="PDF(또는 로컬 전용 .mov)에서 글자를 추출해 책 PDF 생성")
    ap.add_argument("input", type=Path, help="입력 .pdf (로컬 전용: .mov)")
    ap.add_argument("--out", type=Path, default=Path("output"), help="출력 폴더 (기본: output)")
    ap.add_argument("--work", type=Path, default=None, help="중간 결과 폴더 (기본: <out>/work). 같은 폴더로 재실행하면 이어서 진행")
    ap.add_argument("--name", default="책.pdf", help="출력 PDF 파일명 (기본: 책.pdf)")
    ap.add_argument("--title", default=None, help="표지 제목 (기본: 입력 파일명)")
    ap.add_argument("--dpi", type=int, default=200, help="PDF 페이지 렌더 DPI")
    ap.add_argument("--skip-ocr-if-text", action="store_true", help="텍스트 레이어가 있는 PDF 페이지는 OCR 생략")
    ap.add_argument("--threshold", type=float, default=None,
                    help="신뢰도 낮음 기준 (기본: gemini 0.7, tesseract 0.6 — tesseract 기준값은 실제 책으로 미검증)")
    ap.add_argument("--ocr", choices=["gemini", "tesseract"], default="gemini",
                    help="글자 인식 엔진. gemini=API(일일 한도 있음), tesseract=로컬(kor+eng, 한도 없음, 이어하기 지원)")
    ap.add_argument("--pages", default=None, help="처리할 쪽 범위(PDF는 원본 쪽 번호, 영상은 추출된 장 번호). 예: 114-215 또는 1-3,10")
    ap.add_argument("--embed-page-images", action="store_true",
                    help="[tesseract] 글자가 적고 그림이 많은 쪽은 원본 쪽 이미지를 그대로 책에 삽입(제목만 글자로 유지)")
    ap.add_argument("--heading-ratio", type=float, default=1.25,
                    help="[tesseract] 글자 높이가 본문의 이 배수 이상이면 중제목(##), 1.2배 이상이면 대제목(#). 제목이 과다하면 올리고 빠지면 내리세요 (기본 1.25)")
    ap.add_argument("--embed-min-chars", type=int, default=120, help="[tesseract] 이 글자 수 미만인 쪽을 '글자가 적은 쪽'으로 봄 (기본 120)")
    ap.add_argument("--embed-min-ink", type=float, default=0.04, help="[tesseract] 배경과 다른 화소 비율이 이 이상이어야 삽입 (기본 0.04, 빈 쪽 제외)")
    ap.add_argument("--fps", type=float, default=3.0, help="[.mov 로컬 전용] 초당 추출 프레임 수 (2~5 권장)")
    ap.add_argument("--page-thr", type=float, default=0.10,
                    help="[.mov 로컬 전용] 페이지 구분 민감도(0~1). 같은 페이지가 쪼개지면 올리고, 다른 페이지가 합쳐지면 내리세요")
    ap.add_argument("--spread", choices=["auto", "on", "off"], default="auto",
                    help="펼침면(좌/우 두 쪽이 한 장) 분리: auto=가로세로비+제본선 자동 판별(기본), on=모두 분리, off=분리 안 함")
    ap.add_argument("--spread-thr", type=float, default=spread.DEFAULT_THR,
                    help="펼침면 판정 임계값(0~1, 기본 0.5). 프레임별 점수가 이 값 이상이면 분리. 두 쪽이 한 장으로 남으면 낮추고(0.4, 0.3), 단일 쪽이 잘리면 올리세요")
    ap.add_argument("--split-only", action="store_true",
                    help="OCR·책 PDF 없이 펼침면 분리와 쪽 이미지 저장만 수행하고, 프레임별 판정 로그·미리보기·분리점검.md 를 만든다(API 키·tesseract 불필요)")
    ap.add_argument("--gutter-offset", type=float, default=0.0,
                    help="제본선 위치 수동 보정(이미지 폭 비율, 예: 0.02=오른쪽으로 2%%, -0.02=왼쪽으로 2%%)")
    ap.add_argument("--spread-overlap", type=float, default=0.015,
                    help="분리 시 안쪽으로 겹쳐 자르는 폭(이미지 폭 비율, 기본 0.015). 글자 잘림 방지용")
    ap.add_argument("--max-frame-side", type=int, default=2000, help="[.mov 로컬 전용] 프레임 긴 변 상한(px). 줄이면 메모리·업로드 감소")
    ap.add_argument("--max-calls", type=int, default=0,
                    help="이번 실행에서 Gemini 호출 최대 횟수(0=환경변수 GEMINI_RPD, 그것도 없으면 무제한). 도달하면 저장 후 멈춤")
    ap.add_argument("--allow-partial", action="store_true", help="중단돼 미처리 쪽이 남아도 처리된 쪽만으로 책 PDF 생성")
    ap.add_argument("--restart", action="store_true", help="이전 중간 결과(work)를 지우고 처음부터")
    ap.add_argument("--max-pages", type=int, default=0, help="앞쪽 N페이지만 처리(테스트용)")
    a = ap.parse_args(argv)

    inp: Path = a.input
    sel: list[int] | None = None
    if a.pages:
        try:
            sel = pagesel.parse_pages(a.pages)
        except ValueError as e:
            print(f"[입력 오류] {e}", file=sys.stderr)
            return 2
    if not 0.0 <= a.spread_thr <= 1.0:
        print(f"[입력 오류] --spread-thr 는 0~1 사이여야 합니다(입력: {a.spread_thr}).", file=sys.stderr)
        return 2
    if a.ocr == "tesseract" and not a.split_only:
        from . import ocr_tesseract
        msg = ocr_tesseract.check_available("kor+eng")
        if msg:
            print(f"[tesseract 사용 불가] {msg}", file=sys.stderr)
            return 2
    out: Path = a.out
    work: Path = a.work or (out / "work")
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    s = load_settings()
    extra: list[str] = []
    pdf_for_text: Path | None = None

    # ---- 0. 설정 서명: 설정이 바뀐 채 이어하면 이전 결과와 섞이므로 막는다
    sig = {"input": inp.name, "size": inp.stat().st_size if inp.is_file() else None,
           "mtime": inp.stat().st_mtime_ns if inp.is_file() else None,
           "spread": a.spread, "gutter_offset": a.gutter_offset, "spread_overlap": a.spread_overlap,
           "spread_thr": a.spread_thr, "spread_algo": spread.ALGO_VERSION,
           "ocr": a.ocr, "pages": a.pages, "dpi": a.dpi, "fps": a.fps, "page_thr": a.page_thr, "max_frame_side": a.max_frame_side}
    meta_path = work / "meta.json"
    if a.restart and work.exists():
        shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        # --split-only 는 원본 이미지(입력·dpi·fps 등)만 이전 결과와 맞으면 된다: 분리·OCR 설정은 별도 폴더에 쓰므로 자유롭게 바꿔 시험할 수 있다.
        keys = SRC_KEYS if (a.split_only or old.get("_partial")) else tuple(sig)
        diff = sorted(k for k in keys if old.get(k) != sig[k])
        if diff:
            print(f"[설정 불일치] 이전 실행과 달라진 항목: {', '.join(diff)}\n"
                  f"  이전 결과와 섞이지 않도록 중단합니다. 처음부터 하려면 --restart, 새 폴더를 쓰려면 --work 를 지정하세요."
                  + ("\n  (spread_algo/spread_thr 는 이 버전에서 펼침면 판정 방식이 바뀌어 새로 생긴 항목입니다. 이전 판정으로 만든 쪽 이미지·OCR 결과와 섞지 않으려는 것이니 --restart 를 쓰세요.)"
                     if {"spread_algo", "spread_thr"} & set(diff) else ""),
                  file=sys.stderr)
            return 2
        if old.get("_partial") and not a.split_only:
            meta_path.write_text(json.dumps(sig, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        meta_path.write_text(json.dumps({**sig, **({"_partial": True} if a.split_only else {})}, ensure_ascii=False, indent=1),
                             encoding="utf-8")

    # ---- 1. 입력 검증 + 원본 이미지 만들기
    ext = inp.suffix.lower()
    if ext == ".pdf":
        try:
            n = pdf_input.validate_pdf(inp)
        except ValueError as e:
            print(f"[입력 오류] {e}", file=sys.stderr)
            return 2
        print(f"입력 PDF 확인: {n}쪽")
        if sel is not None and sel[-1] > n:
            print(f"[입력 오류] --pages {a.pages} 가 전체 {n}쪽을 넘습니다.", file=sys.stderr)
            return 2
        sources = pdf_input.render_pages(inp, work / "source_images", a.dpi, sel,
                                         max_side=3600 if a.ocr == "tesseract" else pdf_input.MAX_SIDE)
        pdf_for_text = inp
    elif ext in (".mov", ".mp4", ".m4v"):
        try:
            from .local import video_input  # 로컬 전용(opencv/ffmpeg 필요)
        except ImportError as e:
            print(f"[영상 입력은 로컬 전용] 의존성이 없습니다: {e}\n  pip install -r requirements-video.txt 와 ffmpeg 설치가 필요합니다.", file=sys.stderr)
            return 2
        try:
            sources, vinfo = video_input.extract_pages(inp, work / "source_images", fps=a.fps, same_thr=a.page_thr,
                                                       max_side=a.max_frame_side)
        except (RuntimeError, ValueError) as e:
            print(f"[영상 처리 오류] {e}", file=sys.stderr)
            return 2
        extra += ["## 영상 입력 정보", ""] + [f"- {k}: {v}" for k, v in vinfo.items()]
        if sel is not None:
            if sel[-1] > len(sources):
                print(f"[입력 오류] --pages {a.pages} 가 추출된 {len(sources)}장을 넘습니다.", file=sys.stderr)
                return 2
            sources = [sources[i - 1] for i in sel]
    else:
        print(f"[입력 오류] 지원하지 않는 형식입니다: {ext} (.pdf, 로컬 전용 .mov)", file=sys.stderr)
        return 2

    # ---- 1b. 펼침면 분리 (단일 쪽 이미지는 그대로 통과)
    # --split-only 는 work/split_only 에 따로 쓴다(매번 새로 만듦): 임계값을 바꿔 반복 시험해도 본 실행의 쪽 이미지·OCR 결과를 건드리지 않는다.
    split_dir = work / "split_only" if a.split_only else work
    if a.split_only and split_dir.exists():
        shutil.rmtree(split_dir)
    images, spread_info, frames = spread.expand_spreads_ex(
        sources, split_dir / "page_images", a.spread, a.gutter_offset, a.spread_overlap, thr=a.spread_thr, verbose=a.split_only)
    n_split = sum(1 for s in spread_info if s["kind"] == "left")
    print(f"원본 {len(sources)}장 → 책 {len(images)}쪽 (펼침면 {n_split}장 분리, --spread {a.spread}, --spread-thr {a.spread_thr})")
    spread_notes = spread_report.summary_lines(frames, a.spread_thr, a.spread)
    if a.split_only:
        sheets = spread_report.contact_sheets(frames, sources, split_dir / "preview", a.spread_thr)
        chk = out / "분리점검.md"
        spread_report.write_check_md(chk, frames, a.spread_thr, a.spread, sheets, split_dir / "page_images",
                                     split_dir / "page_images" / "spread_log.csv", len(images))
        print(f"[분리만] 쪽 이미지 {split_dir / 'page_images'} / 판정 로그 {split_dir / 'page_images' / 'spread_log.csv'} / "
              f"미리보기 {len(sheets)}장 {split_dir / 'preview'} / 점검 문서 {chk}")
        return 0
    print("\n".join(l for l in spread_notes if l.startswith("-") or l.startswith("###"))[:1500])
    extra += spread_notes
    pdf_map = sel if (pdf_for_text is not None and sel is not None) else None
    if pdf_for_text is not None and n_split:
        a.skip_ocr_if_text = False  # 분리된 쪽은 PDF 쪽 번호와 대응하지 않으므로 텍스트 레이어를 쓸 수 없다
    if a.max_pages:
        images = images[: a.max_pages]

    # ---- 2. OCR (Gemini) / 텍스트 레이어
    ocr = None
    max_calls = 0
    if a.ocr == "tesseract":
        from . import ocr_tesseract
        ocr = ocr_tesseract.make_tesseract_ocr("kor+eng")
        print("tesseract(kor+eng) 사용: 호출 한도 없음")
    elif s.api_key:
        from .ocr import make_gemini_ocr
        ocr = make_gemini_ocr(s.api_key, s.model, s.rpm)
        print(f"Gemini 사용: 모델 {s.model}, RPM {s.rpm}")
        max_calls = a.max_calls or s.rpd
        hint = pipeline.read_quota_hint(work)
        if hint:
            line = f"지난 실행 기록: {hint['at']} {ocr_kind_label(hint['kind'])} 도달 (그 전까지 성공 {hint['successful_calls']}회)."
            if hint["kind"] == "daily" and hint["successful_calls"] and not max_calls:
                line += f" 하루 한도 안에서 미리 멈추려면 --max-calls {hint['successful_calls']} (또는 GEMINI_RPD) 를 지정하세요."
            elif hint["kind"] == "daily" and max_calls > hint["successful_calls"] > 0:
                line += f" 지정한 상한({max_calls})이 기록({hint['successful_calls']})보다 큽니다."
            print(line + " 일일 한도는 미국 태평양시 자정에 초기화(공식 문서 기준).")
    else:
        print("GEMINI_API_KEY 가 없습니다(.env 확인). 텍스트 레이어가 있는 PDF 페이지만 처리됩니다.")
    results, stopped = pipeline.process_pages(images, work, ocr, pdf=pdf_for_text, use_textlayer=a.skip_ocr_if_text,
                                              max_calls=max_calls, rpm=s.rpm if (ocr and a.ocr == "gemini") else 0.0,
                                              pdf_page_map=pdf_map)
    pending = len(images) - len(results)

    # ---- 3. 후처리
    fig_map = build_figure_map(results)
    toc_entries = None
    if a.ocr == "tesseract":
        opt = structure.EmbedOptions(a.embed_page_images, a.embed_min_chars, a.embed_min_ink)
        mds, emb, toc_all, sstats = structure.assemble([r for r in results if r.get("engine") == "tesseract"], opt, work / "embed", a.heading_ratio)
        fig_map.update(emb)
        extra += ["## 책 조립(tesseract)", ""] + [f"- {k}: {v}" for k, v in sstats.items()]
        page_mds = mds
        md_pages = [r["page"] for r in results if r.get("engine") == "tesseract"]
    else:
        page_mds = [namespaced_markdown(r) for r in results]
        md_pages = [r["page"] for r in results]
    # 글꼴에 없는 글자(한자·원문자 등)를 PDF 만들기 전에 모아 리포트에 남긴다
    glyph_found = glyphs.scan(list(zip(md_pages, page_mds)))
    glyph_lines = glyphs.format_report(glyph_found)
    body, stats = postprocess.postprocess(page_mds)
    if a.ocr == "tesseract":
        toc_entries = structure.match_toc_to_body(toc_all, body)
    (out / "전체원문.md").write_text(body, encoding="utf-8")
    md_dir = out / "pages_md"
    md_dir.mkdir(exist_ok=True)
    for r in results:
        (md_dir / f"p{r['page']:04d}.md").write_text(r.get("markdown", ""), encoding="utf-8")

    # ---- 4. 책 PDF
    low = report.low_confidence_pages(results, a.threshold if a.threshold is not None else (0.6 if a.ocr == "tesseract" else 0.7))
    if stopped and not a.allow_partial:
        report.write_report(out / "리포트.md", results, low, stats, extra + glyph_lines, spread_info, pending)
        print(f"[중단] 미처리 {pending}쪽이 남아 책 PDF 는 만들지 않았습니다(진행상황은 {work} 에 저장됨).\n"
              f"  같은 명령을 다시 실행하면 이어서 처리합니다. 지금까지 분량만 보려면 --allow-partial.", file=sys.stderr)
        return 5
    if not body.strip():
        print("[결과 오류] 추출된 텍스트가 없어 책 PDF를 만들지 않습니다. (API 키/입력을 확인하세요)", file=sys.stderr)
        report.write_report(out / "리포트.md", results, low, stats, extra + glyph_lines, spread_info, pending)
        return 3
    title = a.title or inp.stem
    html_str = render.build_html(body, fig_map, title, render.default_subtitle(), toc_entries)
    pdf_path = out / a.name
    render.build_pdf(html_str, pdf_path, allowed_dirs=[work])

    # ---- 5. 결과 검증 + 리포트
    report.write_report(out / "리포트.md", results, low, stats, extra + glyph_lines, spread_info, pending)
    problems = _verify(pdf_path, results, bool(glyph_found["cjk"]))
    print(f"완료: {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")
    print(f"신뢰도 낮은 페이지 {len(low)}개 → {out / '리포트.md'}")
    for p in problems:
        print(f"[검증 경고] {p}")
    return 0 if not problems else 4


def ocr_kind_label(kind: str) -> str:
    from .ocr import KIND_LABEL
    return KIND_LABEL.get(kind, kind)


def _verify(pdf: Path, results: list[dict], expect_cjk: bool = False) -> list[str]:
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
    if expect_cjk and not any("Noto" in f for f in fonts):
        issues.append("한자·CJK 기호가 있는데 Noto Sans CJK KR 글꼴이 PDF 에 포함되지 않았습니다(빈칸일 수 있음).")
    doc.close()
    return issues


if __name__ == "__main__":
    raise SystemExit(main())
