"""펼침면 판정 점검용 산출물: 요약 문장, 프레임 미리보기 시트(제본선 표시), 분리점검.md."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import FONT_REGULAR
from .spread import DEFAULT_THR, UNCERTAIN_BAND

SHEET_COLS, SHEET_ROWS, THUMB_H = 4, 3, 300
MISSED_EVIDENCE = 0.25      # 단일 판정이지만 띠/그림자 증거가 이만큼 있으면 '놓쳤을 수 있음'


def _split(frames: list[dict]) -> list[dict]:
    return [f for f in frames if f["decision"] == "split"]


def classify(frames: list[dict], thr: float, mode: str) -> dict[str, list[dict]]:
    """점검이 필요한 프레임: 경계(점수가 임계 근처), 주변 기준으로 분리(prior), 단일이지만 증거가 있는 프레임."""
    out: dict[str, list[dict]] = {"border": [], "prior": [], "maybe_missed": []}
    if mode != "auto":
        return out
    for f in frames:
        if f["score"] == "":
            continue
        if f["prior"]:
            out["prior"].append(f)
        elif abs(f["score"] - thr) <= UNCERTAIN_BAND:
            out["border"].append(f)
        elif f["decision"] == "single" and (f["gap_e"] or 0) >= MISSED_EVIDENCE or \
                f["decision"] == "single" and (f["shadow_e"] or 0) >= MISSED_EVIDENCE:
            out["maybe_missed"].append(f)
    return out


def _fmt(f: dict) -> str:
    sc = f"{f['score']:.2f}" if f["score"] != "" else "-"
    return (f"프레임 {f['frame']} ({f['source']}, {f['width']}x{f['height']}, 점수 {sc}, "
            f"띠 {f['gap']}, 그림자 {f['shadow']}, 제본선 x={f['gutter_x']})")


def summary_lines(frames: list[dict], thr: float, mode: str, limit: int = 15) -> list[str]:
    n, k = len(frames), len(_split(frames))
    lines = ["## 펼침면 판정 요약", "",
             f"- 모드 `--spread {mode}`, 임계값 `--spread-thr {thr}`: 원본 {n}장 중 {k}장 분리, {n - k}장 단일",
             "- 프레임별 점수·사유는 `page_images/spread_log.csv` (엑셀에서 열림)"]
    c = classify(frames, thr, mode)
    for key, title, hint in (
        ("prior", "책 전체의 제본선 위치를 기준으로 분리한 프레임 (자체 증거가 약했음)", "잘못 잘렸다면 `--spread-thr` 를 올리거나 해당 프레임을 확인"),
        ("border", f"경계 프레임 (점수가 임계값 ±{UNCERTAIN_BAND} 이내)", "`--spread-thr` 를 조금 바꾸면 판정이 뒤집힙니다"),
        ("maybe_missed", "단일로 판정했지만 제본 띠/그림자 증거가 있는 프레임", "두 쪽이 들어 있다면 `--spread-thr` 를 낮추세요"),
    ):
        if c[key]:
            lines += ["", f"### {title}: {len(c[key])}장", f"- {hint}"]
            lines += [f"  - {_fmt(f)} → {'분리' if f['decision'] == 'split' else '단일'}" for f in c[key][:limit]]
            if len(c[key]) > limit:
                lines.append(f"  - … 외 {len(c[key]) - limit}장 (CSV 참고)")
    return lines + [""]


def _font(size: int):
    try:
        return ImageFont.truetype(str(FONT_REGULAR), size)
    except OSError:
        return ImageFont.load_default()


def contact_sheets(frames: list[dict], sources: list[Path], out_dir: Path, thr: float = DEFAULT_THR) -> list[Path]:
    """프레임 썸네일 격자(4x3) 이미지. 분리선은 빨강(분리), 주변 기준 분리는 주황, 단일인데 후보 위치가 있으면 회색 점선.
    아이폰에서 바로 볼 수 있도록 JPEG 로 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    per = SHEET_COLS * SHEET_ROWS
    font = _font(15)
    paths: list[Path] = []
    for s0 in range(0, len(frames), per):
        chunk = list(zip(frames[s0:s0 + per], sources[s0:s0 + per]))
        thumbs = []
        for f, src in chunk:
            with Image.open(src) as im:
                im = im.convert("RGB")
                s = THUMB_H / im.height
                th = im.resize((max(1, int(im.width * s)), THUMB_H), Image.BILINEAR)
            d = ImageDraw.Draw(th)
            if f["gutter_x"] != "":
                if f["decision"] == "split":
                    col = (255, 140, 0) if f["prior"] else (230, 20, 20)
                    for x in {f["x_min"], f["x_max"]}:
                        d.line([(x * th.width, 0), (x * th.width, THUMB_H)], fill=col, width=2)
                elif f["score"] != "" and (f["gap_e"] or f["shadow_e"]):
                    x = f["gutter_x"] * th.width
                    for y in range(0, THUMB_H, 14):
                        d.line([(x, y), (x, y + 7)], fill=(150, 150, 150), width=2)
            thumbs.append(th)
        widths = sorted(t.width for t in thumbs)
        cell_w = max(120, widths[len(widths) // 2]) + 8          # 중앙값: 가로 프레임 하나가 섞여도 칸이 넓어지지 않게(아이폰에서 보기 좋게)
        thumbs = [t if t.width <= cell_w - 8 else t.resize((cell_w - 8, max(1, int(t.height * (cell_w - 8) / t.width))), Image.BILINEAR)
                  for t in thumbs]
        sheet = Image.new("RGB", (cell_w * SHEET_COLS, (THUMB_H + 26) * SHEET_ROWS), (30, 30, 30))
        d = ImageDraw.Draw(sheet)
        for i, ((f, _), th) in enumerate(zip(chunk, thumbs)):
            cx, cy = (i % SHEET_COLS) * cell_w, (i // SHEET_COLS) * (THUMB_H + 26)
            sheet.paste(th, (cx + 4, cy + 22))
            sc = f"{f['score']:.2f}" if f["score"] != "" else "-"
            tag = "분리" if f["decision"] == "split" else "단일"
            d.text((cx + 6, cy + 2), f"#{f['frame']} {sc} {tag}{'*' if f['uncertain'] else ''}", fill=(255, 255, 255), font=font)
        p = out_dir / f"preview_{s0 // per + 1:02d}.jpg"
        sheet.save(p, quality=80)
        paths.append(p)
    return paths


def write_check_md(path: Path, frames: list[dict], thr: float, mode: str, sheets: list[Path], pages_dir: Path,
                   csv_path: Path, n_pages: int) -> None:
    lines = ["# 펼침면 분리 점검 (--split-only)", "",
             f"- 원본 {len(frames)}장 → 책 {n_pages}쪽",
             f"- 쪽 이미지: `{pages_dir}` (p0001.png … 좌→우 순서, OCR 은 하지 않았습니다)",
             f"- 프레임별 판정 로그: `{csv_path}`",
             f"- 미리보기: " + (", ".join(f"`{p}`" for p in sheets) if sheets else "없음") +
             "  (빨강=분리선, 주황=주변 프레임 기준 분리, 회색 점선=단일로 판정한 후보 위치, `*`=경계/주변 기준)", ""]
    lines += summary_lines(frames, thr, mode)
    lines += ["## 다음 단계", "",
              "1. 미리보기에서 두 쪽이 한 장으로 남은 프레임이 있으면 `--spread-thr` 를 낮추고(예: 0.4, 0.3), 멀쩡한 단일 쪽이 잘렸으면 올립니다(예: 0.6).",
              "2. 선이 쪽 글자를 자르면 `--gutter-offset`(+는 오른쪽)·`--spread-overlap` 으로 보정합니다.",
              "3. 같은 명령에 `--split-only` 를 빼고 실행하면 이 판정으로 OCR·책 PDF 를 만듭니다. "
              "이전 실행의 결과가 같은 폴더에 있으면 설정 불일치로 멈추니 `--restart` 또는 새 `--work` 를 쓰세요."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
