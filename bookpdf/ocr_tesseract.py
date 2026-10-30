"""Tesseract(kor+eng) 기반 페이지 OCR. 호출 한도가 없고 로컬에서 돈다.

Gemini 와 같은 규약(이미지 경로 → 결과 dict)을 따르되, 책 조립용으로 블록별 위치·글자 크기를 함께 남긴다.
제목 판정·머리글 제거·목차는 모든 쪽을 본 뒤에야 알 수 있으므로 structure.py 가 나중에 처리한다.
"""
from __future__ import annotations

import os
import re
import shutil
import statistics
import subprocess
from pathlib import Path
from typing import Callable

from PIL import Image

EDGE_FRAC = 0.09          # 쪽 위/아래 이 비율 안쪽은 머리글/바닥글 후보 구역
MIN_WORD_CONF = 40        # 글자 수 집계에 포함할 단어 신뢰도(0~100)
_DIGITS_ONLY = re.compile(r"^\W*(\d{1,4})\W*$")
_EDGE_NUM = re.compile(r"^(\d{1,4})\s+\S|\S\s+(\d{1,4})$")


def find_binary() -> str | None:
    """TESSERACT_CMD 환경변수 > PATH 순으로 tesseract 실행 파일을 찾는다."""
    cmd = os.environ.get("TESSERACT_CMD", "").strip()
    if cmd:
        return cmd if Path(cmd).exists() or shutil.which(cmd) else None
    return shutil.which("tesseract")


def check_available(lang: str = "kor+eng") -> str | None:
    """사용할 수 있으면 None, 아니면 사용자에게 보여줄 오류 메시지."""
    exe = find_binary()
    if not exe:
        return ("tesseract 를 찾을 수 없습니다. 설치 후 다시 실행하세요.\n"
                "  Linux: sudo apt install tesseract-ocr tesseract-ocr-kor\n"
                "  macOS: brew install tesseract tesseract-lang\n"
                "  Windows: UB Mannheim 설치본 + 한국어(kor) 언어 데이터, 설치 경로를 TESSERACT_CMD 에 지정 (이 PC 환경에서는 미검증)")
    try:
        out = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"tesseract 실행 실패: {e}"
    have = set(out.stdout.split()) | set(out.stderr.split())
    missing = [l for l in lang.split("+") if l not in have]
    if missing:
        return (f"tesseract 언어 데이터가 없습니다: {', '.join(missing)} "
                f"(설치된 것: {', '.join(sorted(have - {'List', 'of', 'available', 'languages', 'in'})) or '없음'})")
    return None


def parse_tsv(tsv: str) -> tuple[tuple[int, int], list[dict]]:
    """tesseract TSV → ((폭, 높이), 줄 목록). 줄: text, x0,y0,x1,y1, h(글자 높이), conf(평균), block, par."""
    page_size = (0, 0)
    lines: dict[tuple[int, int, int], dict] = {}
    for row in tsv.splitlines()[1:]:
        c = row.split("\t")
        if len(c) < 12:
            continue
        level = int(c[0])
        if level == 1:
            page_size = (int(c[8]), int(c[9]))
            continue
        if level != 5:
            continue
        text = c[11].strip()
        try:
            conf = float(c[10])
        except ValueError:
            continue
        if not text or conf < 0:
            continue
        key = (int(c[2]), int(c[3]), int(c[4]))
        x, y, w, h = int(c[6]), int(c[7]), int(c[8]), int(c[9])
        ln = lines.setdefault(key, {"words": [], "x0": x, "y0": y, "x1": x + w, "y1": y + h,
                                    "hs": [], "confs": [], "block": key[0], "par": key[1]})
        ln["words"].append(text)
        ln["x0"], ln["y0"] = min(ln["x0"], x), min(ln["y0"], y)
        ln["x1"], ln["y1"] = max(ln["x1"], x + w), max(ln["y1"], y + h)
        ln["hs"].append(h)
        ln["confs"].append((conf, len(text)))
    out = []
    for key in sorted(lines):
        ln = lines[key]
        hs = sorted(ln["hs"])
        wsum = sum(n for _, n in ln["confs"]) or 1
        out.append({
            "text": " ".join(ln["words"]), "x0": ln["x0"], "y0": ln["y0"], "x1": ln["x1"], "y1": ln["y1"],
            "h": hs[len(hs) // 2] if hs else 0,              # 중앙값: 두 줄이 붙어 인식된 큰 상자 등 이상치에 덜 흔들림
            "conf": sum(c * n for c, n in ln["confs"]) / wsum,
            "chars": sum(n for c, n in ln["confs"] if c >= MIN_WORD_CONF),
            "block": ln["block"], "par": ln["par"],
        })
    return page_size, out


def build_blocks(lines: list[dict], page_h: int) -> tuple[list[dict], int | None]:
    """줄 → 문단 블록. 위/아래 구역의 '숫자만 있는 줄'은 쪽번호로 빼내고 블록에서 제외한다."""
    page_no: int | None = None
    groups: dict[tuple[int, int], list[dict]] = {}
    for ln in lines:
        groups.setdefault((ln["block"], ln["par"]), []).append(ln)
    blocks = []
    edge_hint: int | None = None
    for key in sorted(groups):
        g = groups[key]
        y0, y1 = min(l["y0"] for l in g), max(l["y1"] for l in g)
        zone = "top" if y1 < page_h * EDGE_FRAC * 1.6 and y0 < page_h * EDGE_FRAC else \
               "bottom" if y0 > page_h * (1 - EDGE_FRAC) else "body"
        text = "\n".join(l["text"] for l in g)
        if zone != "body":
            m = _DIGITS_ONLY.match(text)
            if m and len(g) == 1:
                if page_no is None:
                    page_no = int(m.group(1))
                continue                      # 쪽번호 줄은 본문에서 뺀다
            m = _EDGE_NUM.search(text) if len(g) == 1 else None
            if m and edge_hint is None:
                edge_hint = int(m.group(1) or m.group(2))   # '114 책제목' 형태: 번호만 기록하고 줄은 남김(머리글 제거가 처리)
        blocks.append({
            "text": text, "zone": zone, "y0": y0, "y1": y1,
            "x0": min(l["x0"] for l in g), "x1": max(l["x1"] for l in g),
            "lh": statistics.median(l["h"] for l in g), "lines": len(g),
            "chars": sum(l["chars"] for l in g),
            "conf": sum(l["conf"] * len(l["text"]) for l in g) / (sum(len(l["text"]) for l in g) or 1),
        })
    if page_no is None and edge_hint is not None and 0 < edge_hint < 10000:
        page_no = edge_hint
    return blocks, page_no


def ink_ratio(image_path: Path) -> float:
    """쪽 배경(중앙 밝기)과 뚜렷이 다른 화소의 비율. 빈 쪽≈0, 그림·차트·사진이 많을수록 큼."""
    import numpy as np
    with Image.open(image_path) as im:
        g = im.convert("L")
        g = g.resize((400, max(1, int(g.height * 400 / g.width))))
    a = np.asarray(g, dtype=np.int16)
    bg = int(np.median(a))
    return float((np.abs(a - bg) > 40).mean())


def make_tesseract_ocr(lang: str = "kor+eng", psm: int = 3, timeout: int = 240, dpi: int = 300) -> Callable[[Path], dict]:
    # 이미지 DPI 메타데이터(펼침면 분리·영상 프레임 저장 시 사라짐)에 기대면 같은 픽셀도 결과가 달라진다 → 고정값을 명시(tesseract 권장 300)
    exe = find_binary()
    if not exe:
        raise RuntimeError("tesseract 를 찾을 수 없습니다.")

    def ocr(image_path: Path) -> dict:
        proc = subprocess.run([exe, str(image_path), "stdout", "-l", lang, "--psm", str(psm), "--dpi", str(dpi), "tsv"],
                              capture_output=True, timeout=timeout, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
        if proc.returncode != 0:
            raise RuntimeError(f"tesseract 오류(code {proc.returncode}): {proc.stderr.decode('utf-8', 'replace')[:200]}")
        (W, H), lines = parse_tsv(proc.stdout.decode("utf-8", "replace"))
        if H == 0:
            with Image.open(image_path) as im:
                W, H = im.size
        blocks, page_no = build_blocks(lines, H)
        wsum = sum(len(l["text"]) for l in lines) or 1
        conf = sum(l["conf"] * len(l["text"]) for l in lines) / wsum / 100 if lines else 0.0
        return {
            "markdown": "\n\n".join(b["text"] for b in blocks),
            "blocks": blocks,
            "chars": sum(b["chars"] for b in blocks),
            "ink": round(ink_ratio(image_path), 4),
            "size": [W, H],
            "figures": [],
            "confidence": round(max(0.0, min(1.0, conf)), 3),
            "page_number": page_no,
            "engine": "tesseract",
        }

    return ocr
