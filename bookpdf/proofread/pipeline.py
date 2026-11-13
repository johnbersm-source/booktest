"""교정 파이프라인 조립: 정규화 → 탐지 → 후보 → (LLM) → (이미지 검증) → 판정 → (적용) → 리포트.

판정 원칙(보수적): '근거가 둘 이상 독립적으로 일치할 때만' 자동 채택한다.
  · 이미지 검증을 켠 경우: 이미지 재판독이 후보 중 하나와 일치(원문은 불일치)해야 채택. LLM 제안이 이미지 판독과 다르면 충돌 → 보류.
  · 이미지 검증을 끈 경우(--no-verify): LLM 제안이 책 어휘 후보 또는 두 번째 OCR 과 일치할 때만 채택(숫자·영문 변경은 두 번째 OCR 일치 필요).
  · LLM 도 끈 경우(--no-llm --no-verify): 후보에 독립 출처가 2개 이상이고 점수가 높을 때만 채택.
  · 그 밖에는 모두 보류 → 검수 리포트. 이미지가 원문을 뒷받침하면 오탐(dismissed)으로 기록하고 고치지 않는다.
모든 채택은 llm_edit.validate 의 같은 안전 검사(편집거리·길이·숫자/영문·보호어)를 한 번 더 통과해야 한다.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import apply as ap, candidates as cd, detect as dt, llm_edit as le, normalize as nz, util, verify as vf
from .cache import ResponseCache


@dataclass
class Options:
    work: Path
    out: Path | None = None                 # 기본 <work>/proofread
    corrected: Path | None = None           # 기본 <work>/corrected
    pages: list[int] | None = None
    no_llm: bool = False
    no_verify: bool = False
    apply: bool = False
    review: Path | None = None
    alt_work: Path | None = None
    glossary: Path | None = None
    acronyms: Path | None = None
    rebuild_glossary: bool = False
    use_tsv: bool = True
    english_dict: Path | None = None
    quote_style: str = "straight"
    params: dt.Params = field(default_factory=dt.Params)
    limits: le.Limits = field(default_factory=le.Limits)
    min_adopt_cand_score: float = 0.30      # --no-verify 에서 LLM 제안을 뒷받침하는 책 어휘 후보의 최소 점수
    solo_cand_score: float = 0.70           # --no-llm --no-verify 에서 후보 단독 채택 최소 점수(독립 출처 2개 이상일 때)
    glossary_min_count: int = 5
    glossary_min_pages: int = 3


def _read_dict(path: Path | None) -> set[str] | None:
    if not path:
        return None
    try:
        return {w.strip().lower() for w in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines() if w.strip()}
    except OSError:
        return None


def prepare(opts: Options, log=print):
    """쪽 로드·정규화·보호목록·어휘. 반환: dict(공용 상태)."""
    allp = util.load_pages(opts.work)
    if not allp:
        raise FileNotFoundError(f"{opts.work}/pages 에 pNNNN.json 이 없습니다.")
    sel = util.load_pages(opts.work, opts.pages) if opts.pages else allp
    miss = util.missing_pages(sel, opts.pages)
    if miss:
        raise ValueError(f"--pages 에 work 에 없는 쪽이 있습니다: {miss[:10]}")
    out = opts.out or (opts.work / "proofread")
    out.mkdir(parents=True, exist_ok=True)
    gl_user = opts.glossary or (opts.work / "glossary.txt")
    gl_auto = out / "glossary_auto.txt"
    protect0 = nz.Protect.load(gl_user if Path(gl_user).exists() else None, opts.acronyms)
    norm_all = {p["page"]: nz.normalize_text(p["markdown"], protect0, opts.quote_style) for p in allp}
    if opts.rebuild_glossary or not gl_auto.exists():
        items = nz.collect_glossary([r.text for r in norm_all.values()], opts.glossary_min_count, opts.glossary_min_pages)
        nz.write_glossary(gl_auto, items)
        log(f"자동 용어집 {len(items)}개 → {gl_auto} (검토 후 틀린 항목은 지우세요)")
    protect = nz.Protect.load(gl_user if Path(gl_user).exists() else None, opts.acronyms, gl_auto)
    # 용어집이 늘었으므로 정규화를 보호목록 포함으로 다시 수행(결정적)
    norm_all = {p["page"]: nz.normalize_text(p["markdown"], protect, opts.quote_style) for p in allp}
    vocab = cd.build_vocab([r.text for r in norm_all.values()], protect)
    return {"allp": allp, "sel": sel, "norm": norm_all, "protect": protect, "vocab": vocab, "out": out}


def _line_of(texts: dict[int, list[str]], s: dict) -> str:
    return texts[s["page"]][s["line"] - 1]


def _gate(span: dict, line: str, text: str, protect: nz.Protect, lim: le.Limits, rng: tuple[int, int] | None = None) -> dict:
    """후보/제안 한 건을 llm_edit.validate 와 같은 안전 검사에 통과시킨다(확신도는 1.0 으로 둠)."""
    s0, e0 = rng or (span["col"], span["end"])
    raw = {"original": line[s0:e0], "corrected": text, "confidence": 1.0, "reason": ""}
    return le.validate({**span, "col": s0, "end": e0}, line, raw, lim, protect)


def decide(span: dict, cands: list[dict], llm: list[dict], ver: dict | None, line: str, protect: nz.Protect,
           opts: Options, have_verify: bool, have_llm: bool) -> dict:
    """한 의심 구간의 최종 판정. 반환: {status: adopted|held|dismissed, edit, evidence, why, options}"""
    lim = opts.limits
    valid_llm = [e for e in llm if e["status"] in ("proposed", "held")]
    best_llm = next((e for e in valid_llm if e["status"] == "proposed"), None)
    options: list[dict] = []

    def add_opt(text, start, end, src):
        if text and text != line[start:end] and not any(o["text"] == text for o in options):
            options.append({"text": text, "start": start, "end": end, "before": line[start:end], "source": src})

    for e in valid_llm:
        add_opt(e["corrected"], e["start"], e["end"], "llm")
    for c in cands:
        add_opt(c["text"], span["col"], span["end"], "+".join(c["sources"]))
    options = options[:3]
    res = {"status": "held", "edit": None, "evidence": [], "why": "", "options": options,
           "rejected_llm": [{"code": e["code"], "detail": e["detail"], "original": e.get("original"), "corrected": e.get("corrected")}
                            for e in llm if e["status"] == "rejected"]}

    def adopt(text, rng, evidence):
        g = _gate(span, line, text, protect, lim, rng)
        if g["status"] in ("rejected",):
            res["why"] = f"안전 검사 불통과({g['code']}): {g['detail']}"
            return False
        res.update(status="adopted", edit={"start": g["start"], "end": g["end"], "before": g["original"], "after": g["corrected"]},
                   evidence=evidence, why="")
        return True

    # 띄어쓰기만 바꾸는 제안(글자 순서는 그대로): 내용 환각 위험이 없으므로 이미지 근거 없이도 채택한다.
    # 단, 글자 사이 공백(syllable_split)·두 OCR 불일치 신호가 있는 구간에 한해, 글자(공백 제외)가 한 글자도 안 바뀐 경우만.
    sp_only = next((e for e in valid_llm if e["status"] == "proposed" and e["original"].replace(" ", "") == e["corrected"].replace(" ", "")), None)
    if sp_only is not None and {g["name"] for g in span["signals"]} & {"syllable_split", "ocr_disagree"}:
        if adopt(sp_only["corrected"], (sp_only["start"], sp_only["end"]), ["띄어쓰기만 변경(글자 불변)", f"LLM 확신도 {sp_only['confidence']:.2f}"]):
            return res
        return res
    if ver and ver["verdict"] == "supports_original":
        res.update(status="dismissed", why="이미지 재판독이 원문과 같음(오탐)", evidence=["이미지 재판독 = 원문"])
        return res
    if ver and ver["verdict"] == "confirmed":
        m = ver["matched"]
        src_llm = next((e for e in valid_llm if e["corrected"] == m), None)
        if valid_llm and src_llm is None and best_llm is not None:
            res["why"] = f"충돌: LLM 은 '{best_llm['corrected']}', 이미지 재판독은 '{m}'"
            return res
        rng = (src_llm["start"], src_llm["end"]) if src_llm else (span["col"], span["end"])
        srcs = [o["source"] for o in options if o["text"] == m] or ["후보"]
        ev = ["이미지 재판독 일치", f"후보 출처: {', '.join(srcs)}"]
        if adopt(m, rng, ev):
            return res
        return res
    if have_verify:
        note = {"mismatch": "이미지 재판독이 어떤 후보와도 맞지 않음", "unreadable": "이미지 재판독 불가([판독불가])",
                "no_image": "쪽 이미지 없음", "error": "이미지 검증 실패/중단"}.get(ver["verdict"] if ver else "", "")
        if ver is None:
            note = "검증 대상 아님(후보 없음)" if not options else "검증 결과 없음"
        res["why"] = note or "후보 없음"
        return res
    # ----- 이미지 검증 없음: 독립 근거 2개 필요
    alt = span.get("alt")
    if best_llm is not None:
        numeric = best_llm.get("numeric")
        by_vocab = any(c["text"] == best_llm["corrected"] and c["score"] >= opts.min_adopt_cand_score for c in cands)
        by_alt = alt is not None and best_llm["corrected"] == alt
        if (by_alt) or (by_vocab and not numeric):
            ev = ["LLM 제안"] + (["두 번째 OCR 일치"] if by_alt else []) + (["책 어휘 후보 일치"] if by_vocab else [])
            if adopt(best_llm["corrected"], (best_llm["start"], best_llm["end"]), ev):
                return res
            return res
        res["why"] = "LLM 제안만 있음(독립 근거 부족" + (", 숫자/영문은 두 번째 OCR 일치 필요" if numeric else "") + ")"
        return res
    if not have_llm and cands:
        c = cands[0]
        numeric = util.has_digit(span["token"]) or util.has_digit(c["text"]) or span.get("kind") in ("num", "label")
        if len(c["sources"]) >= 2 and c["score"] >= opts.solo_cand_score and not numeric:
            if adopt(c["text"], (span["col"], span["end"]), [f"독립 출처 {len(c['sources'])}개: {', '.join(c['sources'])}"]):
                return res
            return res
    res["why"] = res["why"] or ("후보만 있음(독립 근거 부족)" if options else "후보 없음")
    return res


def run(opts: Options, caller: Callable[..., str] | None = None, log: Callable[[str], None] = print) -> dict:
    """탐지~판정까지 실행해 decisions 를 out/ 에 저장한다. 적용(corrected/ 쓰기)은 apply_run 에서."""
    t0 = time.monotonic()
    st = prepare(opts, log)
    out: Path = st["out"]
    protect, vocab = st["protect"], st["vocab"]
    params = opts.params
    if opts.english_dict:
        params.english_dict = _read_dict(opts.english_dict)
    progress = {"stages": {}}

    def mark(stage):
        progress["stages"][stage] = time.strftime("%H:%M:%S")
        util.write_json(out / "progress.json", progress)

    sel = st["sel"]
    pages_in = [{"page": p["page"], "text": st["norm"][p["page"]].text, "page_number": p.get("page_number")} for p in sel]
    texts = {p["page"]: util.lines_of(p["text"]) for p in pages_in}
    # 두 번째 OCR
    alt_texts = None
    if opts.alt_work:
        alts = util.load_pages(opts.alt_work, opts.pages)
        alt_texts = {a["page"]: nz.normalize_text(a["markdown"], protect, opts.quote_style).text for a in alts}
        log(f"두 번째 OCR 결과 {len(alt_texts)}쪽 불러옴 ({opts.alt_work})")
    # tesseract 단어 신뢰도/상자
    tsv = {}
    for p in sel:
        if p.get("engine") == "tesseract":
            w = dt.load_tsv(p, out / "tsv", run=opts.use_tsv)
            if w:
                tsv[p["page"]] = w
    mark("normalize")
    det = dt.detect_pages(pages_in, vocab, protect, params, alt_texts, tsv)
    spans = det["spans"]
    util.write_json(out / "detections.json", det)
    log(f"탐지: 의심 구간 {len(spans)}건 / {len(pages_in)}쪽 (신호별 {det['stats']['by_signal']}; tsv {len(tsv)}쪽, 두 번째 OCR {len(alt_texts or {})}쪽)")
    mark("detect")
    cands = {s["id"]: cd.gen_candidates(s, vocab) for s in spans}
    util.write_json(out / "candidates.json", cands)
    mark("candidates")

    cache = ResponseCache(out / "cache" / "api_cache.jsonl")
    llm_edits: list[dict] = []
    lst = le.LlmStats()
    have_llm = bool(caller) and not opts.no_llm
    if have_llm and spans:
        log(f"LLM 수정 목록 요청(줄 단위 묶음 {opts.limits.batch_lines}) …")
        llm_edits, lst = le.propose_edits(spans, texts, cands, caller, protect, opts.limits, cache, log=log)
        util.write_json(out / "llm_edits.json", {"edits": llm_edits, "stats": lst.as_dict()})
    mark("llm")
    by_span_llm: dict[str, list[dict]] = {}
    for e in llm_edits:
        if e.get("id"):
            by_span_llm.setdefault(e["id"], []).append(e)

    pages_v = {p["page"]: {"image": p.get("image"), "blocks": p.get("blocks"), "markdown_original": p["markdown"],
                           "lines": texts[p["page"]]} for p in sel}
    have_verify = bool(caller) and not opts.no_verify
    ver: dict[str, dict] = {}
    vst = vf.VerifyStats()
    if have_verify:
        targets, vc = [], {}
        for s in spans:
            texts_c = [e["corrected"] for e in by_span_llm.get(s["id"], []) if e["status"] in ("proposed", "held")] \
                + [c["text"] for c in cands[s["id"]]]
            texts_c = list(dict.fromkeys(texts_c))
            if texts_c:
                targets.append(s)
                vc[s["id"]] = texts_c
        log(f"이미지 근거 검증 {len(targets)}건 …")
        ver, vst = vf.verify_spans(targets, vc, pages_v, caller, out / "crops", cache, log=log)
        util.write_json(out / "verify.json", {"results": ver, "stats": vst.as_dict()})
    mark("verify")

    decisions = []
    for s in spans:
        line = _line_of(texts, s)
        d = decide(s, cands[s["id"]], by_span_llm.get(s["id"], []), ver.get(s["id"]), line, protect, opts, have_verify, have_llm)
        decisions.append({"id": s["id"], "page": s["page"], "line": s["line"], "col": s["col"], "end": s["end"], "token": s["token"],
                          "kind": s["kind"], "score": s["score"], "signals": s["signals"], "context": s["context"], "box": s["box"],
                          "alt": s.get("alt"), "candidates": cands[s["id"]], "verify": ver.get(s["id"]), **d})
    _resolve_overlaps(decisions)
    # 보류 항목에 검수용 번호와 잘라낸 이미지 연결
    hn = 0
    for d in decisions:
        if d["status"] == "held":
            hn += 1
            d["hid"] = f"H{hn:03d}"
            v = d.get("verify")
            if v and v.get("crop"):
                d["crop"] = v["crop"]
            else:
                _, name, mode = vf.make_crop(pages_v.get(d["page"]), {"id": d["id"], "page": d["page"], "line": d["line"],
                                                                         "col": d["col"], "end": d["end"], "box": d["box"]}, out / "crops")
                d["crop"] = name
    stats = summarize(opts, st, det, lst, vst, decisions, caller)
    stats["elapsed_sec"] = round(time.monotonic() - t0, 1)
    util.write_json(out / "decisions.json", {"decisions": decisions, "page_notes": det["page_notes"],
                                             "norm_sha1": {str(p): ap.sha1(r.text) for p, r in st["norm"].items()}})
    util.write_json(out / "stats.json", stats)
    mark("decide")
    return {"decisions": decisions, "stats": stats, "state": st, "detections": det, "llm": llm_edits, "texts": texts}


def _resolve_overlaps(decisions: list[dict]) -> None:
    """같은 줄에서 채택된 수정끼리 범위가 겹치면 점수가 낮은 쪽을 보류로 돌린다."""
    by_line: dict[tuple[int, int], list[dict]] = {}
    for d in decisions:
        if d["status"] == "adopted":
            by_line.setdefault((d["page"], d["line"]), []).append(d)
    for ds in by_line.values():
        ds.sort(key=lambda d: -d["score"])
        kept: list[dict] = []
        for d in ds:
            e = d["edit"]
            if any(e["start"] < k["edit"]["end"] and k["edit"]["start"] < e["end"] for k in kept):
                d.update(status="held", edit=None, why="같은 줄의 다른 채택 수정과 범위가 겹침")
            else:
                kept.append(d)


def summarize(opts: Options, st: dict, det: dict, lst: le.LlmStats, vst: vf.VerifyStats, decisions: list[dict], caller) -> dict:
    cnt = Counter(d["status"] for d in decisions)
    rej = Counter()
    for d in decisions:
        for r in d.get("rejected_llm", []):
            rej[r["code"]] += 1
    unattached = lst.rejected - sum(rej.values())
    if unattached > 0:
        rej["알수없는_id"] += unattached
    norm_rules = Counter()
    norm_lines = 0
    for p in st["sel"]:
        for c in st["norm"][p["page"]].changes:
            norm_lines += 1
            for r in c["rules"]:
                norm_rules[r] += 1
    return {
        "pages": len(st["sel"]), "pages_total_in_work": len(st["allp"]),
        "detected": len(decisions), "by_signal": det["stats"]["by_signal"],
        "adopted": cnt["adopted"], "held": cnt["held"], "dismissed": cnt["dismissed"],
        "rejected_proposals": {"total": sum(rej.values()), "by_code": dict(rej)},
        "llm": lst.as_dict(), "verify": vst.as_dict(),
        "api_calls": {"llm_requests": lst.requests, "verify_requests": vst.calls_planned,
                      "total": getattr(caller, "calls", lst.requests + vst.calls_planned) if caller else 0,
                      "cache_hits": lst.cache_hits + vst.cache_hits},
        "normalize": {"lines_changed": norm_lines, "by_rule": dict(norm_rules)},
        "mode": {"llm": bool(caller) and not opts.no_llm, "verify": bool(caller) and not opts.no_verify,
                 "tsv_pages": det["stats"]["tsv_pages"], "alt_pages": det["stats"]["alt_pages"]},
        "page_notes": len(det["page_notes"]),
    }


# ------------------------------------------------------------------ 적용
def final_edits(decisions: list[dict], review: dict[str, dict] | None = None) -> dict[int, list[dict]]:
    """채택(auto) + 검수 선택(review) → 쪽별 수정 목록. 검수가 '원문 유지'면 아무것도 안 함."""
    per: dict[int, list[dict]] = {}
    for d in decisions:
        e = None
        if d["status"] == "adopted":
            e = {**d["edit"], "source": "auto", "evidence": d["evidence"]}
        elif d["status"] == "held" and review and d.get("hid") in review:
            r = review[d["hid"]]
            if r["action"] == "keep":
                continue
            if r["action"] == "choose":
                o = next((o for o in d["options"] if o["text"] == r["text"]), None)
                rng = (o["start"], o["end"]) if o else (d["col"], d["end"])
            else:
                rng = (d["col"], d["end"])
            line = d["context"]
            e = {"start": rng[0], "end": rng[1], "before": line[rng[0]:rng[1]], "after": r["text"], "source": f"review:{r['action']}",
                 "evidence": ["사용자 검수"]}
        if e is None or e["before"] == e["after"]:
            continue
        per.setdefault(d["page"], []).append({"id": d["id"], "line": d["line"], **e})
    return per


def apply_run(opts: Options, state: dict, decisions: list[dict], review: dict[str, dict] | None = None, log=print) -> dict:
    """corrected/ 에 교정본 쓰기(+수정 로그). 원본 work/pages 는 건드리지 않는다."""
    corrected = opts.corrected or (opts.work / "corrected")
    per = final_edits(decisions, review)
    log_obj = ap.new_log({"pages": opts.pages, "no_llm": opts.no_llm, "no_verify": opts.no_verify,
                          "review": bool(review), "quote_style": opts.quote_style})
    texts = {}
    n_edits = 0
    for p in state["sel"]:
        n = p["page"]
        nr = state["norm"][n]
        edits = per.get(n, [])
        new = ap.apply_edits_to_text(nr.text, edits) if edits else nr.text
        texts[n] = new
        n_edits += len(edits)
        log_obj["pages"][str(n)] = ap.build_page_log(p["markdown"], nr.text, nr.changes, edits, new)
    ap.write_corrected(opts.work, corrected, state["sel"], texts, log_obj)
    log(f"교정본 {len(texts)}쪽 → {corrected} (수정 {n_edits}건 + 표기 정규화). 원본 {opts.work}/pages 는 그대로입니다.")
    return {"corrected_dir": corrected, "edits": n_edits, "pages": len(texts)}
