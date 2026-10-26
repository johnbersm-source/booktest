"""검증 리포트: 신뢰도 낮은 페이지 목록 + 결과 검증."""
from __future__ import annotations

import re
from collections import Counter
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


def check_page_numbers(results: list[dict]) -> dict:
    """인쇄된 쪽번호(page_number)와 처리된 쪽 순번을 비교해 누락·중복·역행을 찾는다.

    - 누락 의심: 순번은 1칸 늘었는데 쪽번호는 2 이상 늘어남 → 그 사이 쪽이 영상 감지에서 빠졌을 가능성
    - 중복 의심: 같은 번호가 두 번, 또는 순번이 더 많이 늘었는데 번호는 덜 늘어남 → 같은 쪽이 두 번 캡처됐을 가능성
    - 역행: 번호가 줄어듦(순서 오류·다른 장 시작 등)
    쪽번호가 없는 쪽(표지·그림 전용 쪽)은 건너뛰고 앞뒤 번호로만 비교한다.
    """
    seq = [(r["page"], r.get("page_number")) for r in results if r.get("page_number") is not None]
    out: dict = {"detected": len(seq), "total": len(results), "missing": [], "duplicates": [], "reversed": [],
                 "offset": None, "usable": len(seq) >= 3}
    if not out["usable"]:
        return out
    out["offset"] = Counter(n - p for p, n in seq).most_common(1)[0][0]  # 가장 흔한 (번호 - 순번)
    for (pa, na), (pb, nb) in zip(seq, seq[1:]):
        di, dn = pb - pa, nb - na
        if dn == di:
            continue
        if dn < 0:
            out["reversed"].append({"after_page": pa, "pages": (pa, pb), "numbers": (na, nb)})
        elif dn == 0:
            out["duplicates"].append({"pages": (pa, pb), "number": na})
        elif dn > di:
            miss = list(range(na + di, nb)) if di == 1 else list(range(na + 1, nb))
            out["missing"].append({"between_pages": (pa, pb), "numbers": (na, nb), "missing_numbers": miss[: dn - di]
                                   if di > 1 else miss, "count": dn - di})
        else:  # 0 < dn < di
            out["duplicates"].append({"pages": (pa, pb), "numbers": (na, nb), "extra": di - dn})
    return out


def format_page_number_report(chk: dict) -> list[str]:
    lines = ["", "## 쪽번호 점검", ""]
    if not chk["usable"]:
        lines.append(f"- 쪽번호를 읽은 쪽이 {chk['detected']}/{chk['total']}개뿐이라 점검하지 못했습니다.")
        return lines
    lines.append(f"- 쪽번호를 읽은 쪽: {chk['detected']}/{chk['total']}  (가장 흔한 번호-순번 차이: {chk['offset']})")
    if not (chk["missing"] or chk["duplicates"] or chk["reversed"]):
        lines.append("- 번호가 이어지는 구간에서 누락·중복·역행은 발견되지 않았습니다.")
        return lines
    for m in chk["missing"]:
        a, b = m["between_pages"]
        nums = ", ".join(map(str, m["missing_numbers"])) or "?"
        lines.append(f"- **누락 의심**: p{a}(번호 {m['numbers'][0]}) 와 p{b}(번호 {m['numbers'][1]}) 사이에 {m['count']}쪽 빠짐 (번호 {nums})")
    for d in chk["duplicates"]:
        a, b = d["pages"]
        if "number" in d:
            lines.append(f"- **중복 의심**: p{a} 와 p{b} 가 같은 번호({d['number']})")
        else:
            lines.append(f"- **중복 의심**: p{a}→p{b} 는 {b - a}쪽 진행인데 번호는 {d['numbers'][0]}→{d['numbers'][1]} ({d['extra']}쪽 더 많음)")
    for d in chk["reversed"]:
        a, b = d["pages"]
        lines.append(f"- **번호 역행**: p{a}(번호 {d['numbers'][0]}) → p{b}(번호 {d['numbers'][1]})")
    lines.append("- 확인 방법: 해당 쪽 원본 이미지는 `work/page_images/` 에 있습니다. 번호 인식 오류일 수도 있으니 이미지를 직접 확인하세요.")
    return lines


def write_report(path: Path, results: list[dict], low: list[dict], pp_stats: dict, extra: list[str],
                 spreads: list[dict] | None = None, pending: int = 0) -> None:
    n = len(results)
    ok = sum(1 for r in results if r.get("markdown", "").strip())
    lines = ["# 변환 리포트", "", f"- 전체 페이지: {n}", f"- 텍스트 추출 성공: {ok}",
             f"- 신뢰도 낮음/실패 페이지: {len(low)}", f"- 후처리: {pp_stats}", ""]
    if pending:
        lines += [f"- **미처리 {pending}쪽**: 일일 호출 한도 등으로 중단됨. 같은 명령을 다시 실행하면 이어서 처리합니다.", ""]
    if spreads:
        n_sp = sum(1 for s in spreads if s["kind"] == "left")
        lines += ["## 펼침면 분리", "", f"- 원본 {len({s['source'] for s in spreads})}장 → 책 {len(spreads)}쪽 (펼침면 {n_sp}장 분리)", ""]
    lines += extra
    lines += format_page_number_report(check_page_numbers(results))
    lines += ["", "## 신뢰도 낮은 페이지 (원본 확인 권장)", ""]
    lines += [f"- p{x['page']}: " + "; ".join(x["reasons"]) for x in low] or ["- 없음"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
