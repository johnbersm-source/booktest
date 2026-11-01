"""신뢰도 기준 미만 쪽만 다른 모델(GEMINI_MODEL)로 다시 읽는다 (--redo-below).

원칙
- 기존 결과는 지우지 않는다: 바꾸기 전에 work/pages_prev/pNNNN.vK.json(.md) 로 복사한다(v1=최초 결과).
- 새 결과가 실패했거나 비었거나 신뢰도가 기존보다 낮으면 기존 결과를 유지한다(유지 사유를 로그에 남김).
- 같은 모델로 이미 시도한 쪽은 다시 호출하지 않는다(work/redo_state.json). 이어하기·무료 한도 절약용.
- 새 그림 크롭은 work/figures_redo 에 저장해 기존 그림 파일을 덮어쓰지 않는다.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Callable

from .ocr import KIND_LABEL, QuotaExhausted
from .pipeline import crop_figures

REDO_ENGINES = ("tesseract", "gemini")
STATE_NAME = "redo_state.json"


def load_state(work: Path) -> dict[str, dict]:
    p = work / STATE_NAME
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        return {}


def save_state(work: Path, state: dict[str, dict]) -> None:
    (work / STATE_NAME).write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")


def pick_targets(results: list[dict], below: float, model: str, state: dict[str, dict],
                 skip: Callable[[dict], bool] | None = None) -> tuple[list[dict], dict[str, list[int]]]:
    """다시 읽을 쪽 목록과, 건너뛴 쪽의 사유별 목록. 신뢰도가 below 미만인 쪽만 대상."""
    todo: list[dict] = []
    skipped: dict[str, list[int]] = {"이미 이 모델로 시도": [], "그림 쪽(원본 이미지 삽입)": [], "처리 실패 쪽(일반 재실행이 담당)": []}
    for r in results:
        if r.get("engine") not in REDO_ENGINES or r.get("confidence", 0.0) >= below:
            continue
        if r.get("error"):
            skipped["처리 실패 쪽(일반 재실행이 담당)"].append(r["page"])
        elif state.get(str(r["page"]), {}).get("model") == model:
            skipped["이미 이 모델로 시도"].append(r["page"])
        elif skip is not None and skip(r):
            skipped["그림 쪽(원본 이미지 삽입)"].append(r["page"])
        else:
            todo.append(r)
    return todo, {k: v for k, v in skipped.items() if v}


def _backup(res_dir: Path, prev_dir: Path, page: int) -> Path:
    prev_dir.mkdir(parents=True, exist_ok=True)
    k = 1
    while (prev_dir / f"p{page:04d}.v{k}.json").exists():
        k += 1
    dst = prev_dir / f"p{page:04d}.v{k}.json"
    shutil.copy2(res_dir / f"p{page:04d}.json", dst)
    md = res_dir / f"p{page:04d}.md"
    if md.exists():
        shutil.copy2(md, prev_dir / f"p{page:04d}.v{k}.md")
    return dst


def decide(old: dict, new: dict) -> tuple[bool, str]:
    """새 결과로 교체할지와 사유. 신뢰도는 엔진마다 산출 방식이 달라 같은 척도가 아님에 유의(리포트에 명시)."""
    if new.get("error"):
        return False, f"새 결과 오류: {new['error'][:80]}"
    if not new.get("markdown", "").strip():
        return False, "새 결과에 텍스트 없음"
    if new.get("confidence", 0.0) < old.get("confidence", 0.0):
        return False, f"새 신뢰도 {new.get('confidence', 0.0):.2f} < 기존 {old.get('confidence', 0.0):.2f}"
    return True, f"신뢰도 {old.get('confidence', 0.0):.2f} → {new.get('confidence', 0.0):.2f}"


def redo_low_pages(results: list[dict], work: Path, ocr: Callable[[Path], dict], below: float, model: str,
                   log: Callable[[str], None] = print, max_calls: int = 0,
                   skip: Callable[[dict], bool] | None = None, now: Callable[[], str] | None = None) -> dict:
    """results 를 제자리에서 갱신하고 {'tried','replaced','kept','failed','stopped'} 를 돌려준다."""
    now = now or (lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    res_dir, prev_dir = work / "pages", work / "pages_prev"
    state = load_state(work)
    todo, skipped = pick_targets(results, below, model, state, skip)
    out = {"tried": 0, "replaced": 0, "kept": 0, "failed": 0, "stopped": None, "targets": len(todo)}
    for why, pages in skipped.items():
        log(f"재처리 제외({why}): " + ", ".join(f"p{p}" for p in pages[:30]) + (" …" if len(pages) > 30 else ""))
    if not todo:
        log(f"재처리 대상 없음(신뢰도 < {below}, 모델 {model})")
        return out
    log(f"재처리 대상 {len(todo)}쪽 (신뢰도 < {below}, 모델 {model}) — 기존 결과는 {prev_dir} 에 보존, 새 결과가 더 낫지 않으면 기존 유지")
    calls = 0
    for r in todo:
        page = r["page"]
        if max_calls and calls >= max_calls:
            out["stopped"] = f"호출 상한({max_calls}회)에 도달"
            break
        img = Path(r["image"])
        try:
            calls += 1
            new = ocr(img)
        except QuotaExhausted as e:
            out["stopped"] = f"{KIND_LABEL.get(e.kind, e.kind)}: {e}"
            log(f"[재처리 p{page}] {out['stopped']}")
            break
        except Exception as e:  # noqa: BLE001  한 쪽 실패가 전체를 막지 않게(기록하지 않아 다음 실행에서 재시도)
            out["failed"] += 1
            log(f"[재처리 p{page}] 실패, 기존 유지(다음 실행에서 재시도): {str(e)[:120]}")
            continue
        out["tried"] += 1
        ok, why = decide(r, new)
        entry = {"model": model, "at": now(), "from_engine": r.get("engine"), "from_conf": round(r.get("confidence", 0.0), 3),
                 "to_conf": round(new.get("confidence", 0.0), 3), "replaced": ok, "reason": why}
        if ok:
            keep = _backup(res_dir, prev_dir, page)
            new["page"], new["image"] = page, str(img)
            new["figures"] = crop_figures(img, new.get("figures", []), work / "figures_redo", page)
            new["model"] = model
            new["redo"] = {k: entry[k] for k in ("model", "at", "from_engine", "from_conf", "to_conf")}
            (res_dir / f"p{page:04d}.json").write_text(json.dumps(new, ensure_ascii=False, indent=1), encoding="utf-8")
            (res_dir / f"p{page:04d}.md").write_text(new["markdown"], encoding="utf-8")
            results[results.index(r)] = new
            entry["backup"] = keep.name
            out["replaced"] += 1
            log(f"[재처리 p{page}] 교체: {r.get('engine')} → {new.get('engine')}({model}), {why} (이전 결과 보존: {keep.name})")
        else:
            out["kept"] += 1
            log(f"[재처리 p{page}] 기존 유지: {why}")
        state[str(page)] = entry
        save_state(work, state)       # 쪽마다 저장: 중간에 끊겨도 이어서 가능
    if out["stopped"]:
        log(f"재처리 중단: {out['stopped']}. 같은 명령을 다시 실행하면 남은 쪽만 이어서 처리합니다.")
    log(f"재처리 결과: 시도 {out['tried']}쪽 중 교체 {out['replaced']}, 유지 {out['kept']}, 실패 {out['failed']}")
    return out


def report_lines(work: Path) -> list[str]:
    """리포트용: 누적된 재처리 기록(이전 실행분 포함)."""
    state = load_state(work)
    if not state:
        return []
    rows = sorted(state.items(), key=lambda kv: int(kv[0]))
    lines = ["## 재처리 기록 (--redo-below)", "",
             "- 신뢰도는 엔진마다 산출 방식이 달라(tesseract=단어 인식 평균, Gemini=모델 자기 평가) 같은 척도가 아닙니다. 숫자가 올라도 실제 품질은 PDF 에서 확인하세요.",
             f"- 이전 결과 보존 위치: `{work / 'pages_prev'}` (pNNNN.vK.json, v1=최초)",
             "", "| 쪽 | 모델 | 이전 | 이후 | 결과 | 사유 |", "|---|---|---|---|---|---|"]
    for p, e in rows:
        lines.append(f"| p{p} | {e['model']} | {e['from_engine']} {e['from_conf']} | {e['to_conf']} | "
                     f"{'교체' if e['replaced'] else '유지'} | {e['reason']} |")
    return lines + [""]
