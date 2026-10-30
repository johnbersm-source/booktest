"""Markdown → HTML → xhtml2pdf(NanumGothic). WeasyPrint 는 쓰지 않는다."""
from __future__ import annotations

import datetime as dt
import html
import re
from pathlib import Path

import markdown as md
from PIL import Image
from xhtml2pdf import pisa

from .config import FONT_BOLD, FONT_REGULAR

MAX_IMG_W = 440  # A4 본문 폭(pt) 이내
MAX_IMG_H = 600

CSS = """
@font-face {{ font-family: NanumGothic; src: url("{reg}"); }}
@font-face {{ font-family: NanumGothic; font-weight: bold; src: url("{bold}"); }}
@page {{ size: A4; margin: 2.2cm 2cm 2.2cm 2cm;
  @frame footer {{ -pdf-frame-content: footer; bottom: 0.8cm; margin-left: 2cm; margin-right: 2cm; height: 0.8cm; }} }}
body {{ font-family: NanumGothic; font-size: 10.5pt; line-height: 150%; }}
h1 {{ font-size: 20pt; margin-top: 18pt; -pdf-outline: true; -pdf-outline-level: 0; }}
h2 {{ font-size: 15pt; margin-top: 14pt; -pdf-outline: true; -pdf-outline-level: 1; }}
h3 {{ font-size: 12.5pt; margin-top: 10pt; }}
p {{ margin: 0 0 7pt 0; }}
table {{ border: 0.7pt solid #444444; margin: 6pt 0 10pt 0; }}
th {{ background-color: #e8e8e8; font-weight: bold; padding: 4pt; border: 0.5pt solid #777777; }}
td {{ padding: 4pt; border: 0.5pt solid #777777; }}
.fig {{ text-align: center; margin: 8pt 0; }}
.cap {{ font-size: 8.5pt; color: #555555; text-align: center; }}
.cover-title {{ font-size: 32pt; font-weight: bold; text-align: center; margin-top: 230pt; }}
.cover-sub {{ font-size: 12pt; text-align: center; color: #555555; margin-top: 20pt; }}
.toc-item {{ margin: 0 0 4pt 0; }}
.toc-l2 {{ margin-left: 14pt; font-size: 9.5pt; }}
"""


def _escape_raw_html(text: str) -> str:
    # LLM 출력에 섞인 태그가 xhtml2pdf 를 깨뜨리지 않도록 '<' 만 이스케이프
    return text.replace("<", "&lt;")


def _img_html(path: str, caption: str, frac_w: float = 1.0) -> str:
    """원본 페이지에서 차지하던 폭 비율을 본문 폭에 맞춰 환산(너무 작거나 크지 않게 제한)."""
    with Image.open(path) as im:
        w, h = im.size
    tw = max(90.0, min(MAX_IMG_W, MAX_IMG_W * frac_w))
    th = tw * h / w
    if th > MAX_IMG_H:
        tw, th = tw * MAX_IMG_H / th, MAX_IMG_H
    cap = f'<p class="cap">{html.escape(caption)}</p>' if caption else ""
    return (f'<div class="fig"><img src="{Path(path).resolve().as_posix()}" '
            f'width="{int(tw)}" height="{int(th)}" />{cap}</div>')


def build_html(body_md: str, figures: dict[str, dict], title: str, subtitle: str,
               toc_entries: list[dict] | None = None) -> str:
    """figures: 마커 'FIGKEY' → {path, caption}. 본문의 {{FIG:key}} 를 이미지로 치환한다."""
    tokens: dict[str, str] = {}

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in figures:
            return ""
        tok = f"FIGTOKEN{len(tokens)}X"
        tokens[tok] = _img_html(figures[key]["path"], figures[key].get("caption", ""),
                                figures[key].get("frac_w", 1.0))
        return tok

    body_md = re.sub(r"\{\{FIG:([\w\-]+)\}\}", sub, _escape_raw_html(body_md))
    body = md.markdown(body_md, extensions=["tables", "sane_lists"])
    body = body.replace("<table>", '<table border="1" cellpadding="4" cellspacing="0" repeat="1">')
    for tok, h in tokens.items():
        body = body.replace(f"<p>{tok}</p>", h).replace(tok, h)

    toc = []
    if toc_entries is not None:  # 원본 쪽번호가 붙은 목차(tesseract 모드). 인쇄 쪽번호가 없으면 번호 생략
        for e in toc_entries:
            cls = "toc-item" + (" toc-l2" if e["level"] == 2 else "")
            pn = e.get("printed")
            ref = f' <span class="cap">…… 원본 {"~" if e.get("est") else ""}{pn}쪽</span>' if pn else ""
            toc.append(f'<p class="{cls}">{html.escape(e["text"])}{ref}</p>')
    else:
        for m in re.finditer(r"<h([12])>(.*?)</h\1>", body, re.S):
            cls = "toc-item" + (" toc-l2" if m.group(1) == "2" else "")
            toc.append(f'<p class="{cls}">{re.sub("<.*?>", "", m.group(2))}</p>')
    toc_html = ""
    if toc:
        toc_html = "<h1>목차</h1>" + "".join(toc[:200]) + "<pdf:nextpage />"

    css = CSS.format(reg=FONT_REGULAR.resolve().as_posix(), bold=FONT_BOLD.resolve().as_posix())
    return f"""<html><head><meta charset="utf-8"/><style>{css}</style></head><body>
<div class="cover-title">{html.escape(title)}</div>
<div class="cover-sub">{html.escape(subtitle)}</div>
<pdf:nextpage />
{toc_html}
{body}
<div id="footer" style="font-family: NanumGothic; text-align:center; font-size:8pt; color:#777777;"><pdf:pagenumber /></div>
</body></html>"""


def _policy(allowed_dirs: list[Path]):
    """xhtml2pdf>=0.2.28 은 로컬 파일 읽기를 기준 폴더로 제한한다.
    제한을 끄지 않고, 폰트/그림 폴더만 명시적으로 허용하며 원격 접근은 막는다.
    (구버전은 이 정책 클래스가 없어 None → 기본 동작)"""
    try:
        from xhtml2pdf.config.resources import ResourceAccessPolicy
    except ImportError:
        return None
    roots = [Path(d).resolve() for d in allowed_dirs]
    return ResourceAccessPolicy(allow_remote=False, base_dir=roots[0], extra_roots=tuple(roots[1:]))


def build_pdf(html_str: str, out_pdf: Path, allowed_dirs: list[Path] | None = None) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    dirs = [FONT_REGULAR.parent] + list(allowed_dirs or [])
    kwargs = {}
    pol = _policy(dirs)
    if pol is not None:
        kwargs["resource_policy"] = pol
    with open(out_pdf, "wb") as f:
        res = pisa.CreatePDF(html_str, dest=f, encoding="utf-8", **kwargs)
    if res.err:
        raise RuntimeError(f"xhtml2pdf 변환 오류 {res.err}건")


def default_subtitle() -> str:
    return dt.date.today().strftime("%Y-%m-%d") + " 생성"
