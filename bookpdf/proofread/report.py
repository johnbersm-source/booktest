"""M6 리포트: 수정 전후 diff, 보류 항목(후보 2~3개 + 잘라낸 줄 이미지), 통계를 아이폰에서 읽기 쉬운 HTML 과 Markdown 으로.

HTML: 한 열 카드, 글자 16px 이상, 가로 스크롤 없음, 다크모드 지원, 표 대신 카드. 이미지는 report.html 과 같은 폴더의 crops/ 를 상대 경로로 가리킨다.
※ 리포트에는 책 본문 일부(문맥)가 들어갑니다. 출력 폴더(proofread/)는 .gitignore 대상이며 커밋하지 마세요.
"""
from __future__ import annotations

import html
from pathlib import Path

from . import apply as ap, util

SIGNAL_LABEL = {
    "low_conf": "OCR 신뢰도 낮음", "ocr_disagree": "두 OCR 불일치", "rare_near_freq": "드문 낱말(흔한 낱말과 유사)",
    "year_range": "연도 이상", "caption_seq": "그림/표 번호 이상", "sum_mismatch": "합계 불일치", "english_far": "영문 이상",
    "mixed_script": "한글 속 영문/숫자", "syllable_split": "글자 사이 공백", "unreadable": "[판독불가]", "bad_char": "깨진 글자",
}
STATUS_LABEL = {"adopted": "자동 수정", "held": "보류(검수 필요)", "dismissed": "오탐(수정 안 함)"}


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _sig_names(d: dict) -> str:
    return ", ".join(SIGNAL_LABEL.get(s["name"], s["name"]) for s in d["signals"])


def _ctx_html(d: dict, mode: str) -> str:
    """문맥 줄에서 대상 부분을 표시. mode: diff(삭제/삽입) | mark(강조)."""
    line = d["context"]
    lo = max(0, d["col"] - 30)
    hi = min(len(line), d["end"] + 30)
    pre, mid, post = line[lo:d["col"]], line[d["col"]:d["end"]], line[d["end"]:hi]
    if mode == "diff" and d.get("edit"):
        e = d["edit"]
        pre, mid, post = line[lo:e["start"]], line[e["start"]:e["end"]], line[e["end"]:hi]
        mid_html = f"<del>{_e(mid)}</del><ins>{_e(e['after'])}</ins>"
    else:
        mid_html = f"<mark>{_e(mid)}</mark>"
    return f"{'…' if lo else ''}{_e(pre)}{mid_html}{_e(post)}{'…' if hi < len(line) else ''}"


def _ctx_md(d: dict, mode: str) -> str:
    line = d["context"]
    lo = max(0, d["col"] - 30)
    hi = min(len(line), d["end"] + 30)
    if mode == "diff" and d.get("edit"):
        e = d["edit"]
        return f"{'…' if lo else ''}{line[lo:e['start']]}~~{line[e['start']:e['end']]}~~ **{e['after']}**{line[e['end']:hi]}{'…' if hi < len(line) else ''}"
    return f"{'…' if lo else ''}{line[lo:d['col']]}**[{line[d['col']:d['end']]}]**{line[d['end']:hi]}{'…' if hi < len(line) else ''}"


def stat_lines(stats: dict) -> list[tuple[str, str]]:
    v, l, a = stats["verify"], stats["llm"], stats["api_calls"]
    rows = [
        ("처리한 쪽", f"{stats['pages']}쪽 (work 전체 {stats['pages_total_in_work']}쪽)"),
        ("탐지(의심 구간)", f"{stats['detected']}건"),
        ("자동 수정(채택)", f"{stats['adopted']}건"),
        ("보류(검수 필요)", f"{stats['held']}건"),
        ("오탐 판정(이미지가 원문과 같음)", f"{stats['dismissed']}건"),
        ("거절된 LLM 제안(안전 검사 불통과)", f"{stats['rejected_proposals']['total']}건"),
        ("Gemini 호출", f"{a['total']}회 (텍스트 {a['llm_requests']}, 이미지 {a['verify_requests']}, 캐시 재사용 {a['cache_hits']})"),
        ("표기 정규화", f"{stats['normalize']['lines_changed']}줄 변경"),
    ]
    mode = stats["mode"]
    rows.append(("실행 모드", f"LLM {'켬' if mode['llm'] else '끔'} / 이미지검증 {'켬' if mode['verify'] else '끔'} / "
                            f"tesseract 신뢰도 {mode['tsv_pages']}쪽 / 두 번째 OCR {mode['alt_pages']}쪽"))
    if l.get("stopped"):
        rows.append(("⚠ LLM 중단", str(l["stopped"])))
    if v.get("stopped"):
        rows.append(("⚠ 이미지 검증 중단", str(v["stopped"])))
    if l.get("errors"):
        rows.append(("⚠ LLM 형식 오류", f"{len(l['errors'])}건 건너뜀"))
    return rows


CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--card:#f4f4f6;--line:#d8d8de;--ins:#d6f5dc;--del:#fbd9d9;--mk:#fff1a8;--acc:#0a63d6}
@media (prefers-color-scheme:dark){:root{--bg:#111113;--fg:#ececf0;--mut:#a0a0a8;--card:#1d1d21;--line:#34343a;--ins:#173a22;--del:#4a1f1f;--mk:#5a4a00;--acc:#6aa8ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,"Apple SD Gothic Neo","Noto Sans KR",sans-serif;padding:14px;max-width:720px;margin:0 auto;word-break:keep-all;overflow-wrap:anywhere}
h1{font-size:22px;margin:.3em 0}h2{font-size:18px;margin:1.6em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.2em}
.card{background:var(--card);border-radius:12px;padding:12px 14px;margin:10px 0}.mut{color:var(--mut);font-size:14px}
.stat{display:flex;justify-content:space-between;gap:10px;padding:6px 0;border-bottom:1px solid var(--line)}.stat:last-child{border:0}.stat b{white-space:nowrap}
del{background:var(--del);text-decoration:line-through;padding:0 2px;border-radius:3px}ins{background:var(--ins);text-decoration:none;padding:0 2px;border-radius:3px;font-weight:600}
mark{background:var(--mk);color:inherit;padding:0 2px;border-radius:3px}.ctx{font-size:16px;margin:.3em 0}
.chip{display:inline-block;background:var(--line);border-radius:10px;padding:1px 8px;font-size:13px;margin:2px 4px 2px 0}
img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;margin-top:6px}details>summary{cursor:pointer;font-weight:600;padding:6px 0}
.opt{margin:2px 0}code{background:var(--line);padding:0 4px;border-radius:4px}
"""


def render_html(stats: dict, decisions: list[dict], page_notes: list[dict], llm_edits: list[dict], title: str = "교정 리포트") -> str:
    o = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>{_e(title)}</title><style>{CSS}</style></head><body>", f"<h1>{_e(title)}</h1>",
         "<p class='mut'>이 리포트에는 책 본문 일부가 들어 있습니다. 공유·커밋하지 마세요.</p>", "<h2>요약</h2><div class='card'>"]
    for k, v in stat_lines(stats):
        o.append(f"<div class='stat'><span>{_e(k)}</span><b>{_e(v)}</b></div>")
    o.append("</div>")
    if stats["by_signal"]:
        o.append("<div class='card'><div class='mut'>탐지 신호별 건수</div>" + "".join(
            f"<span class='chip'>{_e(SIGNAL_LABEL.get(k, k))} {v}</span>" for k, v in stats["by_signal"].items()) + "</div>")
    rc = stats["rejected_proposals"]["by_code"]
    if rc:
        o.append("<div class='card'><div class='mut'>거절된 LLM 제안 사유</div>" + "".join(
            f"<span class='chip'>{_e(k)} {v}</span>" for k, v in sorted(rc.items(), key=lambda kv: -kv[1])) + "</div>")
    held = [d for d in decisions if d["status"] == "held"]
    adopted = [d for d in decisions if d["status"] == "adopted"]
    dismissed = [d for d in decisions if d["status"] == "dismissed"]

    o.append(f"<h2>보류 항목 {len(held)}건 — 검수 필요</h2>")
    if held:
        o.append("<p class='mut'>검수 파일 <code>held_review.txt</code> 의 같은 번호 줄에 답을 적고 --review 로 다시 실행하면 반영됩니다.</p>")
    for d in held:
        o.append(f"<div class='card'><b>{_e(d['hid'])}</b> · p{d['page']} {d['line']}행 <span class='mut'>점수 {d['score']:.2f}</span>"
                 f"<div class='ctx'>{_ctx_html(d, 'mark')}</div><div>" +
                 "".join(f"<span class='chip'>{_e(SIGNAL_LABEL.get(s['name'], s['name']))}</span>" for s in d["signals"]) + "</div>")
        for i, op in enumerate(d["options"], 1):
            o.append(f"<div class='opt'>{i}) <b>{_e(op['text'])}</b> <span class='mut'>({_e(op['source'])})</span></div>")
        if not d["options"]:
            o.append("<div class='mut'>후보 없음 — 직접 입력(t:글자)</div>")
        o.append(f"<div class='mut'>보류 이유: {_e(d['why'] or '근거 부족')}</div>")
        if d.get("crop"):
            o.append(f"<a href='crops/{_e(d['crop'])}'><img loading='lazy' src='crops/{_e(d['crop'])}' alt='{_e(d['hid'])} 원본 줄 이미지'></a>")
        o.append("</div>")

    o.append(f"<h2>자동 수정 {len(adopted)}건</h2>")
    cur = None
    for d in sorted(adopted, key=lambda d: (d["page"], d["line"], d["col"])):
        if d["page"] != cur:
            if cur is not None:
                o.append("</details>")
            cur = d["page"]
            n = sum(1 for x in adopted if x["page"] == cur)
            o.append(f"<details><summary>p{cur} — {n}건</summary>")
        o.append(f"<div class='card'><div class='ctx'>{_ctx_html(d, 'diff')}</div><span class='mut'>{d['line']}행 · 근거: {_e(' + '.join(d['evidence']))}</span></div>")
    if cur is not None:
        o.append("</details>")
    if dismissed:
        o.append(f"<h2>오탐 {len(dismissed)}건 (이미지가 원문과 같아 고치지 않음)</h2><details><summary>목록 보기</summary>")
        for d in dismissed:
            o.append(f"<div class='card'>p{d['page']} {d['line']}행 <div class='ctx'>{_ctx_html(d, 'mark')}</div></div>")
        o.append("</details>")
    rej = [e for e in llm_edits if e["status"] == "rejected"]
    if rej:
        o.append(f"<h2>거절된 LLM 제안 {len(rej)}건</h2><details><summary>목록 보기 (최대 50건)</summary>")
        for e in rej[:50]:
            o.append(f"<div class='card'><span class='chip'>{_e(e.get('code'))}</span> {_e(e.get('original'))} → {_e(e.get('corrected'))}"
                     f"<div class='mut'>{_e(e.get('detail'))} {('· ' + e['id']) if e.get('id') else ''}</div></div>")
        o.append("</details>")
    if page_notes:
        o.append("<h2>쪽 단위 노트</h2>")
        for n in page_notes:
            o.append(f"<div class='card'>p{n['page']}: {_e(n['detail'])}</div>")
    o.append("</body></html>")
    return "\n".join(o)


def render_md(stats: dict, decisions: list[dict], page_notes: list[dict], llm_edits: list[dict], title: str = "교정 리포트") -> str:
    o = [f"# {title}", "", "> 책 본문 일부가 들어 있으니 공유·커밋하지 마세요.", "", "## 요약"]
    for k, v in stat_lines(stats):
        o.append(f"- {k}: **{v}**")
    if stats["by_signal"]:
        o += ["", "신호별: " + ", ".join(f"{SIGNAL_LABEL.get(k, k)} {v}" for k, v in stats["by_signal"].items())]
    rc = stats["rejected_proposals"]["by_code"]
    if rc:
        o += ["", "거절 사유: " + ", ".join(f"{k} {v}" for k, v in sorted(rc.items(), key=lambda kv: -kv[1]))]
    held = [d for d in decisions if d["status"] == "held"]
    adopted = [d for d in decisions if d["status"] == "adopted"]
    dismissed = [d for d in decisions if d["status"] == "dismissed"]
    o += ["", f"## 보류 {len(held)}건 (검수 필요)", ""]
    for d in held:
        o.append(f"### {d['hid']} · p{d['page']} {d['line']}행")
        o.append(_ctx_md(d, "mark"))
        o.append("신호: " + _sig_names(d))
        for i, op in enumerate(d["options"], 1):
            o.append(f"{i}. **{op['text']}** ({op['source']})")
        if not d["options"]:
            o.append("후보 없음 — 직접 입력(t:글자)")
        o.append(f"이유: {d['why'] or '근거 부족'}")
        if d.get("crop"):
            o.append(f"![{d['hid']}](crops/{d['crop']})")
        o.append("")
    o += [f"## 자동 수정 {len(adopted)}건", ""]
    for d in sorted(adopted, key=lambda d: (d["page"], d["line"], d["col"])):
        o.append(f"- p{d['page']} {d['line']}행: {_ctx_md(d, 'diff')}  \n  근거: {' + '.join(d['evidence'])}")
    if dismissed:
        o += ["", f"## 오탐 {len(dismissed)}건 (이미지가 원문과 같아 고치지 않음)", ""]
        o += [f"- p{d['page']} {d['line']}행: {_ctx_md(d, 'mark')}" for d in dismissed]
    rej = [e for e in llm_edits if e["status"] == "rejected"]
    if rej:
        o += ["", f"## 거절된 LLM 제안 {len(rej)}건 (최대 50건 표시)", ""]
        o += [f"- [{e.get('code')}] {e.get('original')} → {e.get('corrected')} ({e.get('detail')})" for e in rej[:50]]
    if page_notes:
        o += ["", "## 쪽 단위 노트", ""] + [f"- p{n['page']}: {n['detail']}" for n in page_notes]
    return "\n".join(o) + "\n"


def write_reports(out: Path, stats: dict, decisions: list[dict], page_notes: list[dict], llm_edits: list[dict]) -> dict[str, Path]:
    out = Path(out)
    p_html, p_md, p_rev = out / "report.html", out / "report.md", out / "held_review.txt"
    p_html.write_text(render_html(stats, decisions, page_notes, llm_edits), encoding="utf-8")
    p_md.write_text(render_md(stats, decisions, page_notes, llm_edits), encoding="utf-8")
    ap.write_review_template(p_rev, ap.held_view(decisions))
    return {"html": p_html, "md": p_md, "review": p_rev}
