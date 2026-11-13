"""M7 평가: 손으로 교정한 정답 쪽(gold/pNNNN.md)과 비교해 글자 오류율(CER)·단어 오류율(WER)의 수정 전/후를 계산하고,
채택된 수정 중 무작위 N건을 뽑아 '잘못 고친 비율' 검수표를 만든다.

주의
  · 정답 쪽 수가 적으면(예: 10쪽) 수치는 참고용이다. 리포트에 정답 글자 수·쪽 수를 함께 적는다.
  · 정답 md 는 OCR md 를 손으로 고친 것이어야 한다(같은 줄 구조 권장, 줄이 달라도 CER/WER 는 계산됨).
  · 이 모듈은 책 본문을 어디로도 보내지 않는다. gold/ 는 커밋 금지(.gitignore).
비교 방식: 양쪽 모두 NFC + 줄바꿈/연속 공백을 한 칸으로 정리(canon)한 뒤 편집거리를 잰다. 'CER(공백 무시)' 는 공백을 모두 뺀 값 — 띄어쓰기 차이를 빼고 보려는 보조 지표.
"""
from __future__ import annotations

import json
import random
import re
import unicodedata
from pathlib import Path

from . import apply as ap, util

_FIG = util.FIG_MARK


def canon(s: str) -> str:
    s = _FIG.sub("", unicodedata.normalize("NFC", s))
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str, ignore_space: bool = False) -> tuple[int, int]:
    """(편집거리, 기준 글자 수)."""
    r, h = canon(ref), canon(hyp)
    if ignore_space:
        r, h = r.replace(" ", ""), h.replace(" ", "")
    return util.myers_distance(r, h), len(r)


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = canon(ref).split(), canon(hyp).split()
    return util.myers_distance(r, h), len(r)


def _rate(d: int, n: int) -> float | None:
    return round(d / n, 4) if n else None


def read_gold(gold_dir: Path) -> dict[int, str]:
    out = {}
    for p in Path(gold_dir).glob("p*.md"):
        m = re.fullmatch(r"p(\d+)\.md", p.name)
        if m:
            out[int(m.group(1))] = p.read_text(encoding="utf-8")
    return out


def evaluate(work: Path, corrected: Path, gold_dir: Path) -> dict:
    """work/pages(원본 OCR) · corrected(교정본+로그) · gold 를 쪽별로 비교."""
    work, corrected = Path(work), Path(corrected)
    gold = read_gold(gold_dir)
    log = json.loads((corrected / "corrections_log.json").read_text(encoding="utf-8"))
    orig = {p["page"]: p["markdown"] for p in util.load_pages(work)}
    rows, tot = [], {k: [0, 0] for k in ("before", "norm", "after", "before_ns", "after_ns", "wb", "wa")}
    missing = []
    for n in sorted(gold):
        cp = corrected / f"p{n:04d}.md"
        if n not in orig or not cp.exists() or str(n) not in log["pages"]:
            missing.append(n)
            continue
        after = cp.read_text(encoding="utf-8")
        plog = log["pages"][str(n)]
        normalized = ap.revert_edits_in_text(after, plog["edits"])
        g = gold[n]
        row = {"page": n}
        for key, txt in (("before", orig[n]), ("norm", normalized), ("after", after)):
            d, c = cer(g, txt)
            row[f"cer_{key}"] = _rate(d, c)
            tot[key][0] += d
            tot[key][1] += c
        for key, txt in (("before_ns", orig[n]), ("after_ns", after)):
            d, c = cer(g, txt, True)
            tot[key][0] += d
            tot[key][1] += c
        for key, txt in (("wb", orig[n]), ("wa", after)):
            d, c = wer(g, txt)
            tot[key][0] += d
            tot[key][1] += c
        row["edits"] = len(plog["edits"])
        row["verdict"] = ("개선" if row["cer_after"] < row["cer_before"] else "악화" if row["cer_after"] > row["cer_before"] else "동일")
        rows.append(row)
    ok = [r for r in rows]
    summary = {
        "gold_pages": len(gold), "compared_pages": len(rows), "missing_pages": missing,
        "gold_chars": tot["before"][1], "gold_words": tot["wb"][1],
        "cer_before": _rate(*tot["before"]), "cer_normalized_only": _rate(*tot["norm"]), "cer_after": _rate(*tot["after"]),
        "cer_before_nospace": _rate(*tot["before_ns"]), "cer_after_nospace": _rate(*tot["after_ns"]),
        "wer_before": _rate(*tot["wb"]), "wer_after": _rate(*tot["wa"]),
        "pages_improved": sum(1 for r in ok if r["verdict"] == "개선"), "pages_worse": sum(1 for r in ok if r["verdict"] == "악화"),
        "pages_same": sum(1 for r in ok if r["verdict"] == "동일"),
        "caveat": "정답 쪽 수가 적으면 참고용입니다. 책 전체 정확도의 증거가 아닙니다." if len(rows) < 30 else "",
    }
    return {"summary": summary, "pages": rows, "gold_check": gold_check(log, corrected, gold)}


def gold_check(log: dict, corrected_dir: Path, gold: dict[int, str]) -> dict:
    """채택된 자동 수정을 정답과 휴리스틱으로 대조: 정답 쪽에 '수정 후' 글자가 있으면 옳음, 없으면 틀림(원문이 맞았던 경우는 original_was_right).
    (정답이 있는 쪽의 수정만 대상. 같은 글자가 쪽에 여러 번 나오면 부정확할 수 있는 근사치)"""
    right = wrong = unknown = 0
    wrong_items = []
    for ns, plog in log["pages"].items():
        n = int(ns)
        if n not in gold:
            continue
        g = canon(gold[n])
        for e in plog["edits"]:
            if not str(e.get("source", "")).startswith("auto"):
                continue
            a, b = canon(e["after"]), canon(e["before"])
            if not a:
                unknown += 1
            elif a in g and (b not in g or g.count(a) > g.count(b) or b in a):
                right += 1
            else:                       # 수정 후 글자가 정답에 없음 = 정답과 다르게 고침(원문이 맞았거나 엉뚱한 글자)
                wrong += 1
                wrong_items.append({"page": n, "line": e["line"], "before": e["before"], "after": e["after"],
                                    "original_was_right": b in g})
    judged = right + wrong
    return {"checked": right + wrong + unknown, "right": right, "wrong": wrong, "unknown": unknown,
            "wrong_rate": round(wrong / judged, 4) if judged else None, "wrong_items": wrong_items[:30],
            "note": "휴리스틱 근사치(정답 쪽에 같은 글자가 여러 번 나오면 부정확). 정식 수치는 무작위 검수표(--audit)로 확인하세요."}


# ------------------------------------------------------------------ 무작위 검수표
def sample_audit(corrected: Path, decisions_path: Path | None, n: int, seed: int = 0) -> list[dict]:
    """채택된 자동 수정 중 무작위 n건. 검수자가 o(맞게 고침)/x(잘못 고침)/? 로 표시한다."""
    corrected = Path(corrected)
    log = json.loads((corrected / "corrections_log.json").read_text(encoding="utf-8"))
    crops = {}
    if decisions_path and Path(decisions_path).exists():
        for d in json.loads(Path(decisions_path).read_text(encoding="utf-8"))["decisions"]:
            crops[d["id"]] = (d.get("verify") or {}).get("crop") or d.get("crop")
    pool = []
    for ns, plog in log["pages"].items():
        txt = None
        for e in plog["edits"]:
            if not str(e.get("source", "")).startswith("auto"):
                continue
            if txt is None:
                txt = util.lines_of((corrected / f"p{int(ns):04d}.md").read_text(encoding="utf-8"))
            pool.append({"page": int(ns), "line": e["line"], "before": e["before"], "after": e["after"], "id": e.get("id"),
                         "context": txt[e["line"] - 1], "crop": crops.get(e.get("id")), "evidence": e.get("evidence", [])})
    pool.sort(key=lambda r: (r["page"], r["line"], r["id"] or ""))
    rnd = random.Random(seed)
    pick = rnd.sample(pool, min(n, len(pool)))
    pick.sort(key=lambda r: (r["page"], r["line"]))
    for i, r in enumerate(pick, 1):
        r["aid"] = f"A{i:03d}"
    return pick


def write_audit(out: Path, items: list[dict]) -> dict[str, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# 잘못 고친 비율 검수표", "", f"무작위 {len(items)}건. 원본 쪽 이미지와 비교해 `audit_answers.txt` 에 o/x/? 를 적으세요.", ""]
    for r in items:
        md.append(f"## {r['aid']} · p{r['page']} {r['line']}행")
        md.append(f"{r['before']} → **{r['after']}**")
        md.append(f"문맥: {r['context']}")
        if r.get("crop"):
            md.append(f"![{r['aid']}](crops/{r['crop']})")
        md.append("")
    txt = ["# o=맞게 고침, x=잘못 고침(원문이 맞았거나 다른 글자), ?=판단 불가. `A001 = o` 형식. 저장 후:",
           "#   python -m bookpdf proofread <work 폴더> --audit-score 이파일.txt", ""]
    txt += [f"{r['aid']} =" for r in items]
    p_md, p_txt, p_json = out / "audit.md", out / "audit_answers.txt", out / "audit.json"
    p_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    p_txt.write_text("\n".join(txt) + "\n", encoding="utf-8")
    util.write_json(p_json, items)
    return {"md": p_md, "answers": p_txt, "json": p_json}


def score_audit(answers_text: str, items: list[dict]) -> dict:
    ids = {r["aid"] for r in items}
    ans = {}
    errs = []
    for no, raw in enumerate(answers_text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = re.match(r"^\s*(A\d{3,})\s*[=:]\s*(\S*)\s*$", raw)
        if not m or m.group(1) not in ids:
            errs.append(f"{no}행 해석 불가: {raw.strip()[:40]}")
            continue
        if m.group(2):
            if m.group(2).lower() not in ("o", "x", "?"):
                errs.append(f"{no}행: o/x/? 중 하나여야 합니다")
            else:
                ans[m.group(1)] = m.group(2).lower()
    right = sum(1 for v in ans.values() if v == "o")
    wrong = sum(1 for v in ans.values() if v == "x")
    n = right + wrong
    # 표본이 작으므로 Wilson 95% 구간을 함께 낸다
    lo = hi = None
    if n:
        z = 1.96
        p = wrong / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
        lo, hi = round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)
    return {"answered": len(ans), "total": len(items), "right": right, "wrong": wrong, "undecided": sum(1 for v in ans.values() if v == "?"),
            "wrong_rate": round(wrong / n, 4) if n else None, "wilson95": [lo, hi], "errors": errs,
            "note": "표본 수가 작으면 구간이 넓습니다. 구간 상한이 허용치를 넘으면 --min-adopt 기준을 올리거나 --no-verify 를 쓰지 마세요."}
