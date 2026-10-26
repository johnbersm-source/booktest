"""펼침면(좌/우 두 쪽이 한 장에 찍힌 이미지) 분리.

- 가로세로비로 후보를 거르고, 가운데 대역에서 '글자 없는 띠 + 제본 그림자'로 제본선(gutter)을 찾는다.
- 위/중간/아래 3구간에서 따로 찾아, 책이 기울었거나 제본선이 휘어도 글자가 잘리지 않게
  왼쪽은 가장 오른쪽 제본선까지, 오른쪽은 가장 왼쪽 제본선부터 + 여백(overlap)만큼 겹쳐 자른다.
- numpy + Pillow 만 사용(PDF 입력 환경에서도 동작).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

MIN_ASPECT = 1.25        # 가로/세로가 이 값 이상이어야 펼침면 후보
BAND = (0.30, 0.70)      # 제본선을 찾는 가로 범위(이미지 폭 비율)
MIN_GAP_FRAC = 0.02      # 글자 없는 띠의 최소 폭(이미지 폭 비율). 이보다 좁으면 제본선 못 찾음
WORK_W = 900             # 분석용 축소 폭


@dataclass
class Gutter:
    x: float             # 제본선 위치(이미지 폭 비율, 0~1) — 3구간 중 중간값
    x_min: float         # 3구간 중 가장 왼쪽
    x_max: float         # 3구간 중 가장 오른쪽
    gap: float           # 글자 없는 띠 폭(비율)
    found: bool          # 글자 없는 띠를 실제로 찾았는지
    shadow: float        # 제본 그림자 깊이(0~1)


def _gray_small(img: Image.Image) -> np.ndarray:
    g = img.convert("L")
    h = max(1, int(g.height * WORK_W / g.width))
    return np.asarray(g.resize((WORK_W, h), Image.BILINEAR), dtype=np.float32)


def _ink_mask(g: np.ndarray) -> np.ndarray:
    """국소 대비로 글자 획만 뽑는다(제본 그림자 같은 완만한 명암은 제거)."""
    im = Image.fromarray(g.astype(np.uint8))
    bg = np.asarray(im.filter(ImageFilter.GaussianBlur(12)), dtype=np.float32)
    return (bg - g) > 22


def _smooth(v: np.ndarray, k: int) -> np.ndarray:
    k = max(1, k)
    return np.convolve(v, np.ones(k) / k, mode="same")


def _bg_profile(g: np.ndarray) -> np.ndarray:
    """열별 배경 밝기(80퍼센타일). 글자 획의 영향을 받지 않고 제본 그림자만 드러난다."""
    return np.percentile(g, 80, axis=0)


def _runs(blank: np.ndarray, lo: int, hi: int, merge: int) -> list[tuple[int, int]]:
    """band 안의 '글자 없는' 연속 구간들. 가장자리 표식 등 짧은 끊김(merge 이하)은 이어 붙인다."""
    runs: list[list[int]] = []
    x = lo
    while x < hi:
        if blank[x]:
            s = x
            while x < hi and blank[x]:
                x += 1
            if runs and s - runs[-1][1] <= merge:
                runs[-1][1] = x
            else:
                runs.append([s, x])
        else:
            x += 1
    return [(r[0], r[1]) for r in runs]


def _find_in_rows(g: np.ndarray, ink: np.ndarray, y0: int, y1: int,
                  hint: float | None = None, hint_ok: bool = False) -> tuple[float, float, bool, float]:
    """한 가로띠(y0:y1)에서 (제본선 x 비율, 띠 폭 비율, 찾음 여부, 그림자 깊이).
    제본 그림자가 뚜렷하면 그 최저점이 제본선(전체 이미지 기준 hint 근처에서만 탐색),
    없으면 글자 없는 띠 중 이미지 중앙에 가장 가까운 것의 가운데."""
    W = g.shape[1]
    lo, hi = int(W * BAND[0]), int(W * BAND[1])
    dens = _smooth(ink[y0:y1].mean(axis=0), int(W * 0.012))
    ref = float(np.percentile(dens, 75)) or 1e-6
    runs = _runs(dens < 0.12 * ref, lo, hi, int(W * 0.012))
    bright = _smooth(_bg_profile(g[y0:y1]), int(W * 0.01))
    if hint_ok and hint is not None:
        wl, wh = max(lo, int((hint - 0.06) * W)), min(hi, int((hint + 0.06) * W))
    else:
        wl, wh = lo, hi
    seg = bright[wl:wh]
    sx = wl + int(np.argmin(seg))
    depth = float((np.median(bright[lo:hi]) - bright[sx]) / 255)
    if depth > 0.03:                      # 제본 그림자가 있으면 그 위치가 제본선
        near = [r for r in runs if r[0] - W * 0.02 <= sx <= r[1] + W * 0.02]
        gap = (near[0][1] - near[0][0]) / W if near else 0.0
        return sx / W, gap, gap >= MIN_GAP_FRAC, depth
    # 그림자가 없는 평평한 스캔: 중앙(±1.5%)을 포함하는 글자 없는 띠만 제본선 후보.
    # 좁은 띠(≤12%)는 가운데, 넓은 띠는 쪽 안쪽 여백이 들쭉날쭉한 경우가 많아 중앙(0.5)을 띠 안쪽으로 보정해 사용.
    mid = W / 2
    cand = [r for r in runs if (r[1] - r[0]) >= W * MIN_GAP_FRAC and r[0] <= mid + 0.015 * W and r[1] >= mid - 0.015 * W]
    if not cand:
        return 0.5, 0.0, False, depth
    r = max(cand, key=lambda r: r[1] - r[0])
    w = r[1] - r[0]
    x = (r[0] + r[1]) / 2 if w <= 0.12 * W else min(max(mid, r[0] + 0.1 * w), r[1] - 0.1 * w)
    return x / W, w / W, True, depth


def find_gutter(img: Image.Image) -> Gutter:
    g = _gray_small(img)
    ink = _ink_mask(g)
    H = g.shape[0]
    top, bot = int(H * 0.06), int(H * 0.94)          # 위아래 가장자리(책상 등) 제외
    third = (bot - top) // 3
    # 전체 이미지 기준 그림자 위치(힌트): 그림자가 있으면 구간별 탐색을 그 근처로 제한
    W = g.shape[1]
    lo, hi = int(W * BAND[0]), int(W * BAND[1])
    gb = _smooth(_bg_profile(g[top:bot]), int(W * 0.01))
    hx = lo + int(np.argmin(gb[lo:hi]))
    hint_ok = float((np.median(gb[lo:hi]) - gb[hx]) / 255) > 0.03
    xs, gaps, founds, depths = [], [], [], []
    for k in range(3):
        a = top + k * third
        x, gap, ok, d = _find_in_rows(g, ink, a, a + third, hx / W, hint_ok)
        xs.append(x); gaps.append(gap); founds.append(ok); depths.append(d)
    med = float(np.median(xs))
    xs = [x if abs(x - med) <= 0.04 else med for x in xs]   # 한 구간만 크게 벗어난 값은 이상치로 보고 대체
    return Gutter(x=med, x_min=float(min(xs)), x_max=float(max(xs)),
                  gap=float(np.median(gaps)), found=sum(founds) >= 2, shadow=float(max(depths)))


def is_spread(img: Image.Image, mode: str = "auto") -> tuple[bool, Gutter | None]:
    """mode: auto | on | off. auto 는 가로세로비 + 제본선(글자 없는 띠) 모두 확인."""
    if mode == "off":
        return False, None
    w, h = img.size
    if mode == "on":
        return True, find_gutter(img)
    if w / h < MIN_ASPECT:
        return False, None
    g = find_gutter(img)
    return (g.found and g.gap >= MIN_GAP_FRAC), g


def split_image(img: Image.Image, gutter: Gutter, offset: float = 0.0, overlap: float = 0.015
                ) -> tuple[Image.Image, Image.Image]:
    """왼쪽/오른쪽 이미지. offset: 제본선 수동 보정(폭 비율, +는 오른쪽). overlap: 안쪽으로 겹칠 폭 비율."""
    w, h = img.size
    left_end = min(w, int((gutter.x_max + offset + overlap) * w))
    right_start = max(0, int((gutter.x_min + offset - overlap) * w))
    return img.crop((0, 0, left_end, h)), img.crop((right_start, 0, w, h))


def expand_spreads(images: list[Path], out_dir: Path, mode: str = "auto", offset: float = 0.0,
                   overlap: float = 0.015, log=print) -> tuple[list[Path], list[dict]]:
    """원본 이미지 목록 → 펼침면은 (왼, 오른) 두 장으로 분리한 페이지 목록. 좌→우 순서 유지.
    이미 만들어진 파일은 다시 만들지 않는다(재개용)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info_path = out_dir / "spreads.json"
    pages: list[Path] = []
    infos: list[dict] = []
    n = 0
    for src_idx, p in enumerate(images, start=1):
        with Image.open(p) as im:
            im.load()
            ok, g = is_spread(im, mode)
            if not ok:
                n += 1
                dst = out_dir / f"p{n:04d}{p.suffix}"
                if not dst.exists():
                    im.save(dst, quality=95) if p.suffix.lower() in (".jpg", ".jpeg") else im.save(dst)
                pages.append(dst)
                infos.append({"page": n, "source": src_idx, "kind": "single"})
                continue
            left, right = split_image(im, g, offset, overlap)
            for side, part in (("left", left), ("right", right)):
                n += 1
                dst = out_dir / f"p{n:04d}{p.suffix}"
                if not dst.exists():
                    part.save(dst, quality=95) if p.suffix.lower() in (".jpg", ".jpeg") else part.save(dst)
                pages.append(dst)
                infos.append({"page": n, "source": src_idx, "kind": side, "gutter": asdict(g)})
            if not g.found:
                log(f"  [경고] 원본 {src_idx}번: 제본선(글자 없는 띠)을 못 찾아 중앙/그림자 기준으로 분할 — --gutter-offset 으로 보정하세요")
    info_path.write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    return pages, infos
