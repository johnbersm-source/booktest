"""python -m bookpdf proofread <work 폴더> [옵션]  — OCR 결과 교정 하위 명령.

기본 동작: 탐지 → 후보 → (LLM) → (이미지 검증) → 판정 → 리포트. 교정본(corrected/)은 --apply 를 줘야 만든다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from .. import pagesel, structure
from . import apply as ap, detect as dt, evaluate as ev, gemini, llm_edit as le, pipeline as pl, report as rp, typeset, util


def build_parser() -> argparse.ArgumentParser:
    a = argparse.ArgumentParser(prog="bookpdf proofread", description="OCR 결과(work/pages)를 보수적으로 교정: 근거 있는 수정만 적용, 나머지는 검수 리포트로 보류")
    a.add_argument("work", type=Path, help="python -m bookpdf 가 만든 work 폴더 (pages/pNNNN.json 이 있는 곳)")
    a.add_argument("--pages", default=None, help="처리할 쪽 범위 (work 의 pNNNN 번호 기준). 예: 1-30 또는 5,8-10")
    a.add_argument("--out", type=Path, default=None, help="리포트·중간 결과 폴더 (기본 <work>/proofread)")
    a.add_argument("--corrected", type=Path, default=None, help="교정본 폴더 (기본 <work>/corrected). 원본 work/pages 는 수정하지 않음")
    a.add_argument("--no-llm", action="store_true", help="Gemini 텍스트 수정 목록(M4)을 쓰지 않음")
    a.add_argument("--no-verify", action="store_true", help="쪽 이미지 근거 검증(M5)을 쓰지 않음")
    a.add_argument("--apply", action="store_true", help="채택된 수정을 corrected/ 에 적용(원본은 보존, 수정 로그 기록)")
    a.add_argument("--review", type=Path, default=None, help="검수 파일(held_review.txt)을 읽어 보류 항목에 반영하고 적용 (이전 실행의 decisions.json 필요, API 호출 없음)")
    a.add_argument("--revert", action="store_true", help="corrected/ 를 수정 로그로 원문 복원해 <work>/restored 에 씀")
    a.add_argument("--alt-work", type=Path, default=None, help="두 번째 OCR 결과 work 폴더(예: tesseract 본 + gemini 본). 단어 정렬 불일치를 의심 신호로 사용")
    a.add_argument("--glossary", type=Path, default=None, help="용어집 파일(한 줄에 용어 하나). 기본: <work>/glossary.txt")
    a.add_argument("--acronyms", type=Path, default=None, help="추가 영문 약어 화이트리스트 파일")
    a.add_argument("--rebuild-glossary", action="store_true", help="자동 수집 용어집(glossary_auto.txt)을 다시 만듦")
    a.add_argument("--english-dict", type=Path, default=None, help="(선택) 영어 단어 목록 파일(한 줄 하나, 예: /usr/share/dict/words). 있으면 사전에 없는 영단어도 의심")
    a.add_argument("--no-tsv", action="store_true", help="tesseract 를 다시 돌려 단어 신뢰도/상자를 얻지 않음(캐시된 tsv 는 사용)")
    a.add_argument("--quote-style", choices=["straight", "keep"], default="straight", help="따옴표 통일 방식 (기본 straight: “ ” → \")")
    g = a.add_argument_group("임계값 (기본값은 합성 데이터 기준이며 실제 책으로 미검증)")
    g.add_argument("--conf-thr", type=float, default=60.0, help="tesseract 단어 신뢰도(0~100) 미만이면 의심 (60)")
    g.add_argument("--min-score", type=float, default=0.30, help="의심 점수 하한 (0.30). 올리면 구간이 줄고 놓치는 오류가 늘어남")
    g.add_argument("--rare-max", type=int, default=1, help="책 안에서 이 횟수 이하로 나온 낱말만 '드문 낱말'로 봄 (1)")
    g.add_argument("--year-range", default="1900-2035", help="허용 연도 범위 (1900-2035)")
    g.add_argument("--max-char-dist", type=int, default=3, help="LLM/후보 수정의 글자 편집거리 상한 (3)")
    g.add_argument("--max-jamo-dist", type=int, default=4, help="자모 편집거리 상한 (4)")
    g.add_argument("--min-confidence", type=float, default=0.7, help="LLM 확신도 미만이면 보류 (0.7)")
    g.add_argument("--batch-lines", type=int, default=8, help="Gemini 요청 하나에 담는 줄 수 (8)")
    g.add_argument("--max-calls", type=int, default=0, help="이번 실행의 Gemini 호출 상한 (0=GEMINI_RPD, 없으면 무제한). 도달하면 저장 후 멈춤")
    e = a.add_argument_group("평가 (M7)")
    e.add_argument("--evaluate", type=Path, default=None, metavar="GOLD_DIR", help="정답 쪽 폴더(gold/pNNNN.md)와 비교해 CER/WER 계산")
    e.add_argument("--audit", type=int, default=0, metavar="N", help="채택된 수정 중 무작위 N건의 검수표 생성")
    e.add_argument("--audit-seed", type=int, default=0)
    e.add_argument("--audit-score", type=Path, default=None, metavar="FILE", help="채워진 audit_answers.txt 로 잘못 고친 비율 계산")
    p = a.add_argument_group("책 PDF (M8)")
    p.add_argument("--pdf", action="store_true", help="교정본(corrected/)으로 책 PDF 생성 (xhtml2pdf + NanumGothic)")
    p.add_argument("--from-md", type=Path, default=None, help="Markdown 파일을 쪽 조립 없이 바로 책 PDF 로 조판")
    p.add_argument("--pdf-out", type=Path, default=Path("output_proofread"), help="PDF 출력 폴더")
    p.add_argument("--name", default="책_교정.pdf")
    p.add_argument("--title", default=None)
    p.add_argument("--embed-page-images", action="store_true", help="[tesseract] 원래 실행에서 쓴 옵션과 같게")
    p.add_argument("--embed-min-chars", type=int, default=120)
    p.add_argument("--embed-min-ink", type=float, default=0.04)
    p.add_argument("--heading-ratio", type=float, default=1.25)
    return a


def _opts(a, work: Path) -> pl.Options:
    try:
        y0, y1 = (int(x) for x in a.year_range.split("-"))
    except ValueError:
        raise ValueError(f"--year-range 형식 오류: '{a.year_range}' (예: 1900-2035)")
    sel = pagesel.parse_pages(a.pages) if a.pages else None
    return pl.Options(
        work=work, out=a.out, corrected=a.corrected, pages=sel, no_llm=a.no_llm, no_verify=a.no_verify, apply=a.apply,
        review=a.review, alt_work=a.alt_work, glossary=a.glossary, acronyms=a.acronyms, rebuild_glossary=a.rebuild_glossary,
        use_tsv=not a.no_tsv, english_dict=a.english_dict, quote_style=a.quote_style,
        params=dt.Params(conf_thr=a.conf_thr, min_score=a.min_score, rare_max=a.rare_max, year_min=y0, year_max=y1),
        limits=le.Limits(max_char_dist=a.max_char_dist, max_jamo_dist=a.max_jamo_dist, min_confidence=a.min_confidence, batch_lines=a.batch_lines))


def _print_summary(stats: dict, out: Path, paths: dict, log=print) -> None:
    log("── 교정 요약 ──")
    for k, v in rp.stat_lines(stats):
        log(f"{k}: {v}")
    log(f"리포트: {paths['html']}  (아이폰: 파일 앱에서 열기 / 같은 내용 {paths['md'].name})")
    if stats["held"]:
        log(f"보류 {stats['held']}건 → 검수 파일 {paths['review']} 에 답을 적고 --review 로 반영")


def main(argv: list[str] | None = None, caller: Callable[..., str] | None = None, log: Callable[[str], None] = print) -> int:
    a = build_parser().parse_args(argv)
    work: Path = a.work
    if not (work / "pages").is_dir() and not a.from_md and not a.audit_score:
        print(f"[입력 오류] {work}/pages 가 없습니다. python -m bookpdf 로 만든 work 폴더를 지정하세요.", file=sys.stderr)
        return 2
    try:
        opts = _opts(a, work)
    except ValueError as e:
        print(f"[입력 오류] {e}", file=sys.stderr)
        return 2
    if a.min_confidence < 0 or a.min_confidence > 1 or not 0 <= a.min_score <= 1:
        print("[입력 오류] --min-confidence/--min-score 는 0~1 사이여야 합니다.", file=sys.stderr)
        return 2
    corrected = opts.corrected or (work / "corrected")
    out = opts.out or (work / "proofread")

    # ---- 단독 동작: 되돌리기 / 검수표 채점 / Markdown 조판
    if a.revert:
        restored = work / "restored"
        ok = ap.revert_dir(corrected, restored)
        bad = [p for p, v in ok.items() if not v]
        log(f"원문 복원 {sum(ok.values())}/{len(ok)}쪽 → {restored}" + (f" (실패 {bad}: {restored / 'revert_errors.txt'})" if bad else ""))
        return 0 if not bad else 4
    if a.audit_score:
        items = json.loads((out / "audit.json").read_text(encoding="utf-8"))
        sc = ev.score_audit(a.audit_score.read_text(encoding="utf-8"), items)
        util.write_json(out / "audit_score.json", sc)
        log(f"검수 {sc['answered']}/{sc['total']}건 응답: 맞음 {sc['right']}, 틀림 {sc['wrong']}, 판단불가 {sc['undecided']}")
        log(f"잘못 고친 비율: {sc['wrong_rate']} (95% 구간 {sc['wilson95']}) — {sc['note']}")
        for e in sc["errors"]:
            log(f"  [검수 파일 오류] {e}")
        return 0
    if a.from_md:
        try:
            r = typeset.build_from_md(a.from_md, a.pdf_out / a.name, a.title or a.from_md.stem, [work], log)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"[조판 오류] {e}", file=sys.stderr)
            return 3
        return 0 if not r["problems"] else 4

    # ---- 검수 반영 (API 호출 없음)
    if a.review:
        dec_path = out / "decisions.json"
        if not dec_path.exists():
            print(f"[입력 오류] {dec_path} 가 없습니다. 먼저 --review 없이 한 번 실행하세요.", file=sys.stderr)
            return 2
        saved = json.loads(dec_path.read_text(encoding="utf-8"))
        st = pl.prepare(opts, log)
        stale = [int(p) for p, h in saved["norm_sha1"].items() if int(p) in st["norm"] and ap.sha1(st["norm"][int(p)].text) != h]
        if stale:
            print(f"[입력 오류] 원본 쪽이 지난 실행 이후 바뀌었습니다({stale[:10]}). 다시 실행해 decisions 를 새로 만드세요.", file=sys.stderr)
            return 2
        st["sel"] = [p for p in st["sel"] if str(p["page"]) in saved["norm_sha1"]] if not opts.pages else st["sel"]
        held = ap.held_view(saved["decisions"])
        answers, errs = ap.parse_review(a.review.read_text(encoding="utf-8"), held)
        for e in errs:
            log(f"[검수 파일 경고] {e}")
        r = pl.apply_run(opts, st, saved["decisions"], answers, log)
        log(f"검수 반영: 선택 {len(answers)}건 (원문 유지 {sum(1 for v in answers.values() if v['action'] == 'keep')}) / "
            f"미응답 보류 {len(held) - len(answers)}건 / 오류 {len(errs)}건")
        rc = 0
    else:
        # ---- 본 실행
        use_api = not (a.no_llm and a.no_verify)
        if use_api and caller is None:
            s = gemini.load_settings()
            if not s.api_key:
                print("[입력 오류] GEMINI_API_KEY 가 없습니다(.env 확인). 키 없이 쓰려면 --no-llm --no-verify 로 실행하세요(후보·탐지·리포트만 만들고 자동 수정은 거의 없음).", file=sys.stderr)
                return 2
            log(f"Gemini 모델 {s.model} (RPM {s.rpm}). {gemini.SEND_NOTICE}")
            caller = gemini.make_caller(s, a.max_calls, log)
        try:
            res = pl.run(opts, caller if use_api else None, log)
        except (FileNotFoundError, ValueError) as e:
            print(f"[입력 오류] {e}", file=sys.stderr)
            return 2
        paths = rp.write_reports(res["state"]["out"], res["stats"], res["decisions"], res["detections"]["page_notes"], res["llm"])
        _print_summary(res["stats"], res["state"]["out"], paths, log)
        stopped = res["stats"]["llm"].get("stopped") or res["stats"]["verify"].get("stopped")
        rc = 5 if stopped else 0
        if stopped:
            log(f"[중단] {stopped} — 지금까지 결과와 캐시는 저장됐습니다. 같은 명령을 다시 실행하면 이어서 진행합니다(이미 받은 응답은 재사용).")
        if a.apply:
            pl.apply_run(opts, res["state"], res["decisions"], None, log)

    # ---- 평가
    if a.evaluate:
        if not (corrected / "corrections_log.json").exists():
            print(f"[입력 오류] {corrected} 에 교정본이 없습니다. --apply 와 함께 실행하세요.", file=sys.stderr)
            return 2
        r = ev.evaluate(work, corrected, a.evaluate)
        util.write_json(out / "evaluate.json", r)
        s = r["summary"]
        log(f"── 평가 (정답 {s['compared_pages']}쪽, {s['gold_chars']}자) ──")
        log(f"CER: 수정 전 {s['cer_before']} → 정규화만 {s['cer_normalized_only']} → 수정 후 {s['cer_after']}")
        log(f"WER: 수정 전 {s['wer_before']} → 수정 후 {s['wer_after']}  / 쪽 개선 {s['pages_improved']} 악화 {s['pages_worse']} 동일 {s['pages_same']}")
        gc = r["gold_check"]
        log(f"정답 대조(근사): 자동 수정 {gc['checked']}건 중 맞음 {gc['right']} / 틀림 {gc['wrong']} / 불명 {gc['unknown']} → 잘못 고친 비율 {gc['wrong_rate']}")
        if s["caveat"]:
            log(f"※ {s['caveat']}")
    if a.audit:
        items = ev.sample_audit(corrected, out / "decisions.json", a.audit, a.audit_seed)
        paths_a = ev.write_audit(out, items)
        log(f"무작위 검수표 {len(items)}건 → {paths_a['md']} / 답 입력: {paths_a['answers']}")

    # ---- 책 PDF
    if a.pdf:
        try:
            emb = structure.EmbedOptions(a.embed_page_images, a.embed_min_chars, a.embed_min_ink)
            r = typeset.build_book(work, corrected, a.pdf_out / a.name, a.title or work.resolve().parent.name or "교정본", emb, a.heading_ratio, log)
        except (OSError, ValueError, RuntimeError) as e:
            print(f"[조판 오류] {e}", file=sys.stderr)
            return 3
        rc = rc or (4 if r["problems"] else 0)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
