"""M8 CLI 연결·조판 테스트 (합성 데이터, 가짜 Gemini)."""
import json
import subprocess
import sys
from pathlib import Path

import pymupdf as fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import test_proofread_fixtures as fx  # noqa: E402
import test_proofread_pipeline as tp  # noqa: E402
from bookpdf.proofread import cli as pcli  # noqa: E402


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)


@pytest.fixture
def work(tmp_path):
    return fx.make_work(tmp_path, tp.NOISY, blocks=True)


def cmain(args, **kw):
    """tesseract 설치 여부에 따라 결과가 달라지지 않도록 단어 신뢰도(tsv) 경로는 끈다(그 경로는 test_proofread_tesseract.py 가 따로 검증)."""
    return pcli.main(list(args) + ["--no-tsv"], **kw)


def logs():
    buf = []
    return buf, buf.append


def test_subcommand_dispatch_from_dunder_main(tmp_path):
    work = fx.make_work(tmp_path, tp.NOISY)
    r = subprocess.run([sys.executable, "-m", "bookpdf", "proofread", str(work), "--no-llm", "--no-verify", "--pages", "1-2"],
                       capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert r.returncode == 0, r.stderr
    assert "교정 요약" in r.stdout and (work / "proofread" / "report.html").exists()
    # 기존 사용법(positional input)은 그대로 동작: --help 가 proofread 가 아닌 원래 파서
    h = subprocess.run([sys.executable, "-m", "bookpdf", "--help"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    assert "--ocr" in h.stdout and "proofread" not in h.stdout.split("usage:")[1].split("\n")[0]


def test_missing_key_message_and_input_errors(work, tmp_path, capsys):
    assert cmain([str(work)], log=lambda *_: None) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
    assert cmain([str(tmp_path / "없는폴더")]) == 2
    assert cmain([str(work), "--no-llm", "--no-verify", "--pages", "9-8"]) == 2
    assert cmain([str(work), "--no-llm", "--no-verify", "--year-range", "abc"], log=lambda *_: None) == 2
    assert cmain([str(work), "--no-llm", "--no-verify", "--pages", "99"], log=lambda *_: None) == 2
    assert cmain([str(work), "--no-llm", "--no-verify", "--min-score", "2"], log=lambda *_: None) == 2


def test_full_flow_run_apply_review_evaluate_audit_pdf(work, tmp_path):
    gold = tmp_path / "gold"
    gold.mkdir()
    for n, lines in fx.GOLD.items():
        (gold / f"p{n:04d}.md").write_text("\n".join(lines), encoding="utf-8")
    buf, log = logs()
    rc = cmain([str(work), "--apply", "--evaluate", str(gold), "--audit", "2", "--pdf", "--pdf-out", str(tmp_path / "pdf"),
                    "--title", "합성 책"], caller=tp.Scripted(), log=log)
    text = "\n".join(buf)
    assert rc == 0, text
    assert "자동 수정(채택): 4건" in text and "CER: 수정 전" in text and "잘못 고친 비율 0.0" in text
    assert (work / "corrected" / "corrections_log.json").exists() and (work / "proofread" / "audit.md").exists()
    # 교정본으로 만든 PDF: 교정된 낱말이 들어 있고 오류 낱말은 없음
    pdf = tmp_path / "pdf" / "책_교정.pdf"
    doc = fitz.open(pdf)
    t = "".join(p.get_text() for p in doc)
    flat = t.replace(" ", "").replace("\n", "")
    assert "추세가살아있다고" in flat and "삼아있다고" not in flat and "거래량을함께" in flat
    assert any("Nanum" in f[3] for p in doc for f in p.get_fonts())
    assert (work / "corrected" / "전체원문_교정.md").exists()
    # 검수표 채점
    items = json.loads((work / "proofread" / "audit.json").read_text(encoding="utf-8"))
    ans = work / "ans.txt"
    ans.write_text("\n".join(f"{i['aid']} = o" for i in items), encoding="utf-8")
    buf2, log2 = logs()
    assert cmain([str(work), "--audit-score", str(ans)], log=log2) == 0
    assert "잘못 고친 비율: 0.0" in "\n".join(buf2)


def test_review_flow_via_cli_without_api(work):
    buf, log = logs()
    assert cmain([str(work), "--no-llm", "--no-verify"], log=log) == 0
    rev = work / "proofread" / "held_review.txt"
    txt = rev.read_text(encoding="utf-8")
    assert txt.count(" =\n") == 4
    rev.write_text(txt.replace("H001 =\n", "H001 = 1\n").replace("H002 =\n", "H002 = x\n"), encoding="utf-8")
    buf2, log2 = logs()
    assert cmain([str(work), "--review", str(rev)], log=log2) == 0           # 키 없이도 동작
    out = "\n".join(buf2)
    assert "선택 2건" in out and "원문 유지 1" in out and "미응답 보류 2건" in out
    log_ = json.loads((work / "corrected" / "corrections_log.json").read_text(encoding="utf-8"))
    assert sum(len(p["edits"]) for p in log_["pages"].values()) == 1
    # 원본 변경 후 --review 는 거부
    p = work / "pages" / "p0001.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    d["markdown"] += "\n추가"
    p.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    assert cmain([str(work), "--review", str(rev)], log=lambda *_: None) == 2


def test_revert_via_cli(work):
    cmain([str(work), "--apply"], caller=tp.Scripted(), log=lambda *_: None)
    assert cmain([str(work), "--revert"], log=lambda *_: None) == 0
    for n in range(1, 6):
        assert (work / "restored" / f"p{n:04d}.md").read_text(encoding="utf-8") == (work / "pages" / f"p{n:04d}.md").read_text(encoding="utf-8")


def test_quota_stop_exit_code_and_resume(work):
    from bookpdf.ocr import QuotaExhausted

    class Quota:
        calls = 0

        def __call__(self, *a, **k):
            self.calls += 1
            raise QuotaExhausted("일일 한도(RPD) 도달", "daily")
    buf, log = logs()
    assert cmain([str(work)], caller=Quota(), log=log) == 5
    assert "같은 명령을 다시 실행" in "\n".join(buf) and (work / "proofread" / "report.html").exists()
    # 같은 명령을 정상 호출기로 다시 실행하면 이어서 완료
    assert cmain([str(work), "--apply"], caller=tp.Scripted(), log=lambda *_: None) == 0
    assert (work / "corrected" / "p0001.md").read_text(encoding="utf-8").count("살아") == 1


def test_from_md_and_fonts_hanja_circled(tmp_path):
    md = tmp_path / "교정.md"
    md.write_text("# 제목\n\n한자 兵赤黑 와 원문자 ①❶㉠ 가 있다.\n", encoding="utf-8")
    buf, log = logs()
    assert cmain([str(tmp_path), "--from-md", str(md), "--pdf-out", str(tmp_path / "o")], log=log) == 0
    doc = fitz.open(tmp_path / "o" / "책_교정.pdf")
    t = "".join(p.get_text() for p in doc)
    assert "兵赤黑" in t and "(1)" in t and "(가)" in t and "①" not in t       # 한자는 Noto, 원문자는 (1)·(가) 대체
    fonts = {f[3] for p in doc for f in p.get_fonts()}
    assert any("Noto" in f for f in fonts) and any("Nanum" in f for f in fonts)


def test_corrected_hanja_page_to_pdf_gemini_engine(tmp_path):
    work = fx.make_work(tmp_path, {1: ["# 장", "", "兵法 ①번 항목과 거래량 거래량"], 2: ["본문이 이어진다 거래량 거래량 추세 추세 추세"]}, engine="gemini")
    assert cmain([str(work), "--no-llm", "--no-verify", "--apply", "--pdf", "--pdf-out", str(tmp_path / "o")], log=lambda *_: None) == 0
    t = "".join(p.get_text() for p in fitz.open(tmp_path / "o" / "책_교정.pdf"))
    assert "兵法" in t and "(1)번" in t.replace(" ", "")
