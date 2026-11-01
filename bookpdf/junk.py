"""신뢰도가 낮고 글자가 거의 없는 쪽(번짐·빈 쪽·손가락·책상 등 쓰레기 프레임)을 책 PDF 에서 빼기 (--drop-junk).

지우지 않고 '조립에서만 제외'한다: work/pages 의 원본 결과와 pages_md 는 그대로 두고, 제외 목록을 리포트에 남긴다.
오삭제를 막기 위해 아래 쪽은 기준에 해당해도 제외하지 않는다.
- 원본 이미지로 삽입하기로 한 쪽(embedded): 글자가 적고 신뢰도가 낮은 것이 정상
- 제목이 판정된 쪽(장 시작 쪽 등 글자가 적은 정상 쪽)
- Gemini 가 그림 영역을 찾은 쪽(figures 있음)
- 처리 실패(error) 쪽: 쓰레기가 아니라 실패이므로 신뢰도 낮은 쪽 목록에 남겨 재시도하게 함
"""
from __future__ import annotations

import re

DEFAULT_CONF = 0.35
DEFAULT_CHARS = 30
PREVIEW = 30


def count_chars(md: str) -> int:
    """공백·그림 표식을 뺀 글자 수. 조립된(머리글·쪽번호 제거 후) 마크다운 기준."""
    return len(re.sub(r"\s+", "", re.sub(r"\{\{FIG:[^}]*\}\}", "", md or "")))


def find_junk(results: list[dict], md_by_page: dict[int, str], conf: float = DEFAULT_CONF, chars: int = DEFAULT_CHARS,
              protect_pages: set[int] | None = None) -> list[dict]:
    """제외할 쪽 목록: {page, printed, confidence, chars, reason, preview}."""
    protect = protect_pages or set()
    out = []
    for r in results:
        p = r["page"]
        if r.get("error") or r.get("embedded") or r.get("figures") or p in protect:
            continue
        if p not in md_by_page:
            continue
        md = md_by_page[p]
        n = count_chars(md)
        c = float(r.get("confidence", 0.0))
        if c < conf and n < chars:
            out.append({"page": p, "printed": r.get("page_number"), "confidence": round(c, 3), "chars": n,
                        "reason": f"신뢰도 {c:.2f} < {conf} 이고 글자 {n}자 < {chars}자",
                        "preview": re.sub(r"\s+", " ", md).strip()[:PREVIEW]})
    return out


def report_lines(dropped: list[dict], total_pages: int, conf: float, chars: int) -> list[str]:
    lines = ["## 책 PDF 에서 제외한 쪽 (--drop-junk)", "",
             f"- 기준: 신뢰도 < {conf} 이면서 조립된 글자 수 < {chars}자 (`--junk-conf`, `--junk-chars` 로 조정). "
             "원본 결과(`work/pages`, `pages_md`)와 쪽 이미지(`work/page_images`)는 그대로 있습니다.",
             "- 제외하지 않는 쪽: 원본 이미지로 삽입한 쪽, 제목이 있는 쪽, 그림 영역이 있는 쪽, 처리 실패 쪽"]
    if not dropped:
        return lines + ["- 제외한 쪽 없음", ""]
    share = len(dropped) / max(1, total_pages)
    lines.append(f"- 제외 {len(dropped)}쪽 / 전체 {total_pages}쪽 ({share:.0%})")
    if share > 0.2:
        lines.append("- **경고: 전체의 20% 넘게 제외됐습니다. 기준이 과한지 확인하세요(`--junk-conf` 를 낮추거나 `--junk-chars` 를 줄이기).**")
    lines += ["", "| 쪽(순번) | 인쇄 쪽번호 | 신뢰도 | 글자 수 | 이유 | 내용 앞부분 |", "|---|---|---|---|---|---|"]
    for d in dropped:
        prev = d["preview"].replace("|", "\\|") or "(비어 있음)"
        lines.append(f"| p{d['page']} | {d['printed'] if d['printed'] else '-'} | {d['confidence']:.2f} | {d['chars']} | {d['reason']} | {prev} |")
    return lines + [""]
