"""교정본 → 책 PDF. 현재 조판 함수(structure.assemble → postprocess → render.build_html → xhtml2pdf + NanumGothic)를 그대로 재사용한다.

· corrected/ 의 쪽 json(교정된 markdown·blocks 포함)을 읽어 원래 파이프라인(bookpdf.cli)과 같은 순서로 조립한다.
· 한자·원문자(NanumGothic 에 없는 글자)는 render/glyphs 의 기존 처리(Noto Sans CJK KR 대체 글꼴, ①→(1))가 그대로 적용된다.
· --from-md: 이미 만든 Markdown 파일을 쪽 조립 없이 바로 조판한다(다른 세션의 --from-md 와는 별개의 이 하위 명령 전용 옵션).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .. import cli as main_cli, glyphs, postprocess, render, structure
from . import util


def build_book(work: Path, corrected: Path, out_pdf: Path, title: str, embed: structure.EmbedOptions | None = None,
               heading_ratio: float = 1.25, log: Callable[[str], None] = print) -> dict:
    corrected = Path(corrected)
    pages = util.load_pages(corrected, subdir="")
    if not pages:
        raise FileNotFoundError(f"{corrected} 에 교정본 쪽이 없습니다. 먼저 --apply 로 교정본을 만드세요.")
    results = []
    for p in pages:
        r = dict(p["raw"])
        r.setdefault("page", p["page"])
        r["markdown"] = p["markdown"]
        results.append(r)
    embed = embed or structure.EmbedOptions(False, 120, 0.04)
    fig_map = main_cli.build_figure_map(results)
    toc_entries, toc_all = None, []
    tess = [r for r in results if r.get("engine") == "tesseract"]
    if tess:
        mds, emb, toc_all, sstats = structure.assemble(tess, embed, corrected / "embed", heading_ratio)
        fig_map.update(emb)
        it = iter(mds)
        kept = [r for r in results if r.get("engine") in ("tesseract", "gemini")]
        page_mds = [next(it) if r["engine"] == "tesseract" else main_cli.namespaced_markdown(r) for r in kept]
    else:
        sstats = {}
        kept = results
        page_mds = [main_cli.namespaced_markdown(r) for r in results]
    body, stats = postprocess.postprocess(page_mds)
    if tess:
        toc_entries = structure.match_toc_to_body(toc_all, body)
    (corrected / "전체원문_교정.md").write_text(body, encoding="utf-8")
    found = glyphs.scan([(r["page"], m) for r, m in zip(kept, page_mds)])
    html_str = render.build_html(body, fig_map, title, render.default_subtitle(), toc_entries)
    render.build_pdf(html_str, out_pdf, allowed_dirs=[Path(work), corrected])
    problems = main_cli._verify(out_pdf, results, bool(found["cjk"]))
    log(f"책 PDF: {out_pdf} ({Path(out_pdf).stat().st_size // 1024} KB)")
    for pr in problems:
        log(f"[검증 경고] {pr}")
    return {"pdf": out_pdf, "problems": problems, "glyph_report": glyphs.format_report(found), "assemble": sstats, "post": stats,
            "markdown": corrected / "전체원문_교정.md"}


def build_from_md(md_path: Path, out_pdf: Path, title: str, allowed: list[Path] | None = None,
                  log: Callable[[str], None] = print) -> dict:
    body = Path(md_path).read_text(encoding="utf-8")
    if not body.strip():
        raise ValueError(f"{md_path} 가 비어 있습니다.")
    html_str = render.build_html(body, {}, title, render.default_subtitle(), None)
    render.build_pdf(html_str, out_pdf, allowed_dirs=[Path(md_path).parent] + list(allowed or []))
    found = glyphs.scan([(0, body)])
    problems = main_cli._verify(out_pdf, [], bool(found["cjk"]))
    log(f"책 PDF: {out_pdf} ({Path(out_pdf).stat().st_size // 1024} KB)")
    return {"pdf": out_pdf, "problems": problems, "glyph_report": glyphs.format_report(found)}
