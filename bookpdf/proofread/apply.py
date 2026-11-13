"""M6 적용·되돌리기·검수 반영. 원본(work/pages)은 절대 건드리지 않고 corrected/ 에 새로 쓴다.

수정 로그(corrections_log.json)는 쪽마다 '정규화 변경(줄 단위 이전/이후)'과 '수정(줄, 시작, 끝, 이전, 이후)'을 기록해서
원문 복원(revert_page)이 항상 가능하다. 위치(start/end)는 정규화된 줄 기준이며, 같은 줄의 여러 수정은 오른쪽부터 적용한다.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from . import normalize as nz, util

LOG_VERSION = 1


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def apply_edits_to_text(text: str, edits: list[dict]) -> str:
    """edits: {line, start, end, before, after}. 줄 번호는 1부터, 위치는 해당 줄 기준. 겹치거나 불일치하면 ValueError."""
    lines = util.lines_of(text)
    by_line: dict[int, list[dict]] = {}
    for e in edits:
        by_line.setdefault(e["line"], []).append(e)
    for li, es in by_line.items():
        es.sort(key=lambda e: e["start"], reverse=True)
        for a, b in zip(es, es[1:]):
            if b["end"] > a["start"]:
                raise ValueError(f"{li}행의 수정이 겹칩니다: {b} / {a}")
        ln = lines[li - 1]
        for e in es:
            if ln[e["start"]:e["end"]] != e["before"]:
                raise ValueError(f"{li}행 {e['start']}~{e['end']} 가 기록된 원문 '{e['before']}' 과 다릅니다(텍스트가 바뀜).")
            ln = ln[:e["start"]] + e["after"] + ln[e["end"]:]
        lines[li - 1] = ln
    return "\n".join(lines)


def revert_edits_in_text(text: str, edits: list[dict]) -> str:
    """apply_edits_to_text 의 역연산. edits 는 적용할 때 쓴 것(정규화 줄 기준 위치)을 그대로 준다."""
    lines = util.lines_of(text)
    by_line: dict[int, list[dict]] = {}
    for e in edits:
        by_line.setdefault(e["line"], []).append(e)
    for li, es in by_line.items():
        es.sort(key=lambda e: e["start"])
        ln = lines[li - 1]
        for e in es:             # 왼쪽부터 되돌리면 이후 수정의 위치는 정규화 좌표(start) 그대로가 된다
            s = e["start"]
            if ln[s:s + len(e["after"])] != e["after"]:
                raise ValueError(f"되돌릴 수 없음: {li}행 {s} 위치가 기록된 수정문 '{e['after']}' 과 다릅니다.")
            ln = ln[:s] + e["before"] + ln[s + len(e["after"]):]
        lines[li - 1] = ln
    return "\n".join(lines)


def build_page_log(original: str, normalized: str, norm_changes: list[dict], edits: list[dict], corrected: str) -> dict:
    return {"original_sha1": sha1(original), "normalized_sha1": sha1(normalized), "corrected_sha1": sha1(corrected),
            "normalize": norm_changes, "edits": edits}


def revert_page(corrected: str, plog: dict) -> str:
    """수정 로그로 corrected → 원문 복원(검증 포함: 해시가 맞아야 함)."""
    if sha1(corrected) != plog["corrected_sha1"]:
        raise ValueError("교정본이 로그 기록 이후 바뀌었습니다(해시 불일치) — 안전하게 되돌릴 수 없습니다.")
    normalized = revert_edits_in_text(corrected, plog["edits"])
    if sha1(normalized) != plog["normalized_sha1"]:
        raise ValueError("수정 되돌리기 결과가 기록과 다릅니다.")
    original = nz.revert_text(normalized, plog["normalize"])
    if sha1(original) != plog["original_sha1"]:
        raise ValueError("정규화 되돌리기 결과가 원본 해시와 다릅니다.")
    return original


def write_corrected(work: Path, out_dir: Path, pages: list[dict], texts: dict[int, str], log: dict) -> list[Path]:
    """corrected/pNNNN.md + pNNNN.json(원본 json 복사 + markdown/blocks 갱신 + proofread 메타) + corrections_log.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for pg in pages:
        n = pg["page"]
        if n not in texts:
            continue
        raw = json.loads(json.dumps(pg["raw"], ensure_ascii=False))
        new_md = texts[n]
        raw["markdown_before_proofread_sha1"] = sha1(pg["markdown"])
        raw["markdown"] = new_md
        if raw.get("blocks"):
            raw["blocks"] = _blocks_with_text(raw["blocks"], pg["markdown"], new_md)
        raw["proofread"] = {"edits": len(log["pages"].get(str(n), {}).get("edits", [])),
                            "normalized_lines": len(log["pages"].get(str(n), {}).get("normalize", []))}
        (out_dir / f"p{n:04d}.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / f"p{n:04d}.md").write_text(new_md, encoding="utf-8")
        written.append(out_dir / f"p{n:04d}.md")
    util.write_json(out_dir / "corrections_log.json", log)
    return written


def _blocks_with_text(blocks: list[dict], old_md: str, new_md: str) -> list[dict]:
    """줄 수가 같으면(우리 수정은 줄을 늘리거나 줄이지 않는다) 블록 text 도 교정본 줄로 바꿔 조판(structure.assemble)에 반영한다."""
    old_lines, new_lines = util.lines_of(old_md), util.lines_of(new_md)
    if len(old_lines) != len(new_lines):
        return blocks
    from .verify import block_line_map
    lm = block_line_map(blocks)
    if len(lm) != len(new_lines):
        return blocks
    out = [dict(b) for b in blocks]
    per_block: dict[int, list[str]] = {}
    for i, m in enumerate(lm):
        if m:
            per_block.setdefault(m[0], []).append(new_lines[i])
    for bi, ls in per_block.items():
        out[bi]["text"] = "\n".join(ls)
    return out


def new_log(settings: dict) -> dict:
    return {"version": LOG_VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": settings, "pages": {}}


def revert_dir(corrected_dir: Path, restore_dir: Path) -> dict[int, bool]:
    """corrected/ 의 쪽들을 로그로 원문 복원해 restore_dir/pNNNN.md 에 쓴다. 반환: {쪽: 성공 여부}. 실패 사유는 예외 대신 False + restore_dir/revert_errors.txt."""
    corrected_dir, restore_dir = Path(corrected_dir), Path(restore_dir)
    log = json.loads((corrected_dir / "corrections_log.json").read_text(encoding="utf-8"))
    restore_dir.mkdir(parents=True, exist_ok=True)
    ok, errs = {}, []
    for n, plog in sorted(log["pages"].items(), key=lambda kv: int(kv[0])):
        try:
            md = (corrected_dir / f"p{int(n):04d}.md").read_text(encoding="utf-8")
            (restore_dir / f"p{int(n):04d}.md").write_text(revert_page(md, plog), encoding="utf-8")
            ok[int(n)] = True
        except (ValueError, OSError) as e:
            ok[int(n)] = False
            errs.append(f"p{n}: {e}")
    if errs:
        (restore_dir / "revert_errors.txt").write_text("\n".join(errs), encoding="utf-8")
    return ok


# ------------------------------------------------------------------ 보류 항목 검수 파일
HEADER = """# 보류 항목 검수 — 이 파일에서 `Hxxx =` 줄의 `=` 뒤에만 적으세요 (아이폰 메모/텍스트 편집기로 가능).
#   숫자(1~3)      → 그 번호의 후보로 고침
#   x              → 원문 그대로 둠(고치지 않음)
#   t:직접 쓴 글자 → 그 글자로 고침 (숫자만 쓰고 싶을 때도 t: 사용. 예: t:2022)
#   (빈칸)         → 보류 유지
# '#' 로 시작하는 줄은 설명입니다. 저장 후: python -m bookpdf proofread <work 폴더> --review 이파일.txt --apply
"""


def write_review_template(path: Path, held: list[dict]) -> None:
    out = [HEADER]
    for h in held:
        opts = "  ".join(f"{i}) {o}" for i, o in enumerate(h["options"], 1)) or "(후보 없음 — t: 로 직접 입력)"
        ctx = h["context"].replace("\n", " ")
        out.append(f"# {h['hid']}  p{h['page']} {h['line']}행  원문 [{h['original']}]  {h['why']}")
        out.append(f"#   …{ctx[max(0, h['col'] - 25):h['col']]}[{h['original']}]{ctx[h['end']:h['end'] + 25]}…")
        out.append(f"#   {opts}")
        out.append(f"{h['hid']} =")
        out.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(out), encoding="utf-8")


_ANS = re.compile(r"^\s*(H\d{3,})\s*(?:=|→|:|\s)\s*(.*?)\s*$")


def parse_review(text: str, held: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """검수 파일 → ({hid: {"action": "choose"|"keep"|"custom", "text": 고칠 글자}}, 오류 목록). 빈 답은 건너뜀(보류 유지)."""
    by_hid = {h["hid"]: h for h in held}
    out: dict[str, dict] = {}
    errs: list[str] = []
    for no, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = _ANS.match(raw)
        if not m:
            errs.append(f"{no}행: 해석할 수 없는 줄 '{raw.strip()[:40]}'")
            continue
        hid, ans = m.group(1), m.group(2)
        if hid not in by_hid:
            errs.append(f"{no}행: 알 수 없는 항목 {hid}")
            continue
        if ans == "":
            continue
        h = by_hid[hid]
        if ans.lower() in ("x", "ㅌ", "원문"):
            out[hid] = {"action": "keep", "text": h["original"]}
        elif ans.startswith("t:"):
            t = ans[2:].strip()
            if not t or "\n" in t:
                errs.append(f"{no}행: {hid} 직접 입력이 비었습니다.")
            else:
                out[hid] = {"action": "custom", "text": t}
        elif ans.isdigit():
            k = int(ans)
            if 1 <= k <= len(h["options"]):
                out[hid] = {"action": "choose", "text": h["options"][k - 1]}
            else:
                errs.append(f"{no}행: {hid} 번호 {k} 는 후보 {len(h['options'])}개 범위 밖입니다(직접 입력은 t:글자).")
        else:
            errs.append(f"{no}행: {hid} 답 '{ans[:30]}' — 숫자, x, 또는 t:글자 형식이어야 합니다.")
    return out, errs


def held_view(decisions: list[dict]) -> list[dict]:
    """검수 파일 작성·해석에 쓰는 보류 항목 요약."""
    return [{"hid": d["hid"], "page": d["page"], "line": d["line"], "original": d["token"], "col": d["col"], "end": d["end"],
             "context": d["context"], "options": [o["text"] for o in d["options"]], "why": d["why"] or "근거 부족"}
            for d in decisions if d["status"] == "held"]
