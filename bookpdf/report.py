"""검증 리포트: 신뢰도 낮은 페이지 목록 + 결과 검증."""
from __future__ import annotations

import re
from pathlib import Path


def low_confidence_pages(results: list[dict], threshold: float = 0.7) -> list[dict]:
    out = []
    for r in results:
        reasons = []
        text = r.get("markdown", "")
        if r.get("error"):
            reasons.append(f"처리 실패: {r['error'][:120]}")
        elif not text.strip():
            reasons.append("추출된 텍스트 없음")
        if r.get("confidence", 0) < threshold and not r.get("error"):
            reasons.append(f"모델 신뢰도 {r.get('confidence', 0):.2f} < {threshold}")
        if "[판독불가]" in text:
            reasons.append("판독불가 표시 포함")
        if text and text.count("�") > 0:
            reasons.append("깨진 글자(U+FFFD) 포함")
        if text.strip() and len(re.findall(r"[가-힣]", text)) == 0 and len(text) > 200 and r.get("engine") != "textlayer":
            reasons.append("한글이 전혀 없음(언어 확인 필요)")
        if reasons:
            out.append({"page": r["page"], "reasons": reasons})
    return out


def write_report(path: Path, results: list[dict], low: list[dict], pp_stats: dict, extra: list[str]) -> None:
    n = len(results)
    ok = sum(1 for r in results if r.get("markdown", "").strip())
    lines = ["# 변환 리포트", "", f"- 전체 페이지: {n}", f"- 텍스트 추출 성공: {ok}",
             f"- 신뢰도 낮음/실패 페이지: {len(low)}", f"- 후처리: {pp_stats}", ""]
    lines += extra
    lines += ["", "## 신뢰도 낮은 페이지 (원본 확인 권장)", ""]
    lines += [f"- p{x['page']}: " + "; ".join(x["reasons"]) for x in low] or ["- 없음"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
