"""펼침면(좌/우 두 쪽이 한 장에 찍힌 이미지) 분리.

- 프레임마다 '펼침면 점수'(0~1)를 계산하고 임계값(--spread-thr)과 비교한다.
  점수 = 제본 증거(글자 없는 띠 / 제본 그림자) × 좌우 글자량 균형.
  가로 이미지는 띠·그림자 어느 쪽이든 증거가 되고, 세로 이미지(아이폰 세로 영상 프레임 등)는 두 단 편집 쪽과
  구별하기 위해 제본 그림자를 필수 증거로 본다.
- 제본선은 위/중간/아래 3구간에서 따로 찾아, 책이 기울었거나 휘어도 글자가 잘리지 않게
  왼쪽은 가장 오른쪽 제본선까지, 오른쪽은 가장 왼쪽 제본선부터 + 여백(overlap)만큼 겹쳐 자른다.
- 책 전체에서 확실한 프레임들의 제본선 위치(카메라 고정 가정)를 모아, 증거가 약한 프레임
  (그림·차트 쪽처럼 가운데가 비지 않는 쪽)은 그 위치를 확인해 분리한다(prior).
- numpy + Pillow 만 사용(PDF 입력 환경에서도 동작).
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ALGO_VERSION = 2        # 판정 방식이 바뀌면 올린다: 이전 버전으로 만든 work 와 섞이지 않게 설정 서명에 넣는다
MIN_ASPECT = 1.25        # 가로/세로가 이 값 이상이면 '가로 이미지'(기존 규칙: 띠가 있으면 분리)
PORTRAIT_ASPECT = 0.90   # 이 값 이하면 완전한 '세로 이미지'. 사이는 두 방식을 섞어 쓴다
BAND = (0.25, 0.75)      # 제본선을 찾는 가로 범위(이미지 폭 비율)
MIN_GAP_FRAC = 0.02      # 글자 없는 띠의 최소 폭(이미지 폭 비율). 이보다 좁으면 제본선 못 찾음
WORK_W = 900             # 분석용 축소 폭
DEFAULT_THR = 0.5        # --spread-thr 기본값
LEGACY_FLOOR = 0.55      # 가로 이미지 + 띠(≥2%)를 찾으면 최소 이 점수(이전 버전과 같은 판정 보장)
STRONG = 0.65            # prior(책 전체 제본선 위치) 를 만들 때 쓰는 '확실한 프레임'의 최소 점수
UNCERTAIN_BAND = 0.10    # 점수가 임계값 ±이 안이면 경계 프레임으로 표시


@dataclass
class Gutter:
    x: float             # 제본선 위치(이미지 폭 비율, 0~1) — 3구간 중 중간값
    x_min: float         # 3구간 중 가장 왼쪽
    x_max: float         # 3구간 중 가장 오른쪽
    gap: float           # 글자 없는 띠 폭(비율)
    found: bool          # 글자 없는 띠를 실제로 찾았는지
    shadow: float        # 제본 그림자 깊이(0~1)


@dataclass
class Analysis:
    aspect: float
    gutter: Gutter
    side: float                    # min(왼쪽 글자량, 오른쪽 글자량) / 전체 (0~0.5)
    gap_e: float = 0.0             # 띠 증거 0~1
    shadow_e: float = 0.0          # 그림자 증거 0~1
    side_e: float = 0.0            # 좌우 균형 0~1
    score: float = 0.0
    decision: bool = False
    reason: str = ""
    prior: bool = False            # 주변 프레임의 제본선 위치로 분리했는지
    mass: np.ndarray | None = field(default=None, repr=False)     # 열별 글자 획 밀도(전체 높이)
    bright: np.ndarray | None = field(default=None, repr=False)   # 위/중간/아래 3구간의 열별 배경 밝기 프로필 (3, W)

    @property
    def evidence(self) -> float:
        return max(self.gap_e, self.shadow_e)

    @property
    def uncertain(self) -> bool:
        return self.prior


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


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


def _drop_rules(ink: np.ndarray) -> np.ndarray:
    """가로 전체를 가로지르는 선(책상-종이 경계, 표 윗선 등)은 글자가 아니므로 지운다.
    남겨두면 위·아래 가장자리에서 제본선의 '글자 없는 띠'가 끊긴다."""
    rows = ink.mean(axis=1)
    cap = max(0.6, 2.5 * float(np.percentile(rows, 90)))
    out = ink.copy()
    out[rows > cap] = False
    return out


def _text_rows(ink: np.ndarray) -> tuple[int, int]:
    """글자가 있는 세로 범위. 세로 프레임에서 위아래의 책상 구간이 제본선 탐색을 흐리지 않게 한다.
    글자가 거의 없으면(그림만 있는 쪽 등) 위아래 6%를 뺀 전체."""
    H = ink.shape[0]
    rows = _smooth(ink.mean(axis=1), max(1, int(H * 0.01)))
    ref = float(np.percentile(rows, 90))
    if ref > 0:
        idx = np.where(rows > 0.2 * ref)[0]
        if len(idx) and (idx[-1] - idx[0]) >= 0.25 * H:
            pad = int(H * 0.01)
            return max(0, int(idx[0]) - pad), min(H, int(idx[-1]) + pad)
    return int(H * 0.06), int(H * 0.94)


def _valley_depth(bright: np.ndarray, x: int, W: int, lo: int, hi: int) -> float:
    """x 열의 밝기가 양옆(4~12% 떨어진 곳)보다 얼마나 움푹한지(0~1). 제본 그림자는 양옆이 종이 밝기인 좁은 골짜기이고,
    차트·사진의 어두운 면은 한쪽만 어둡거나 양옆도 어두워 골짜기로 계산되지 않는다."""
    a0, a1 = max(0, x - int(W * 0.12)), max(0, x - int(W * 0.04))
    b0, b1 = min(W, x + int(W * 0.04)), min(W, x + int(W * 0.12))
    if a1 - a0 < 3 or b1 - b0 < 3:
        ref = float(np.median(bright[lo:hi]))
    else:
        ref = min(float(np.median(bright[a0:a1])), float(np.median(bright[b0:b1])))
    return (ref - float(bright[x])) / 255


def _find_in_rows(g: np.ndarray, ink: np.ndarray, y0: int, y1: int, center: float,
                  hint: float | None = None, hint_ok: bool = False) -> tuple[float, float, bool, float]:
    """한 가로띠(y0:y1)에서 (제본선 x 비율, 띠 폭 비율, 찾음 여부, 그림자 깊이).
    제본 그림자가 뚜렷하면 그 최저점이 제본선(전체 이미지 기준 hint 근처에서만 탐색),
    없으면 글자 없는 띠 중 중앙(center)을 포함하는 것의 가운데."""
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
    depth = _valley_depth(bright, sx, W, lo, hi)
    if depth > 0.03:                      # 제본 그림자가 있으면 그 위치가 제본선
        near = [r for r in runs if r[0] - W * 0.02 <= sx <= r[1] + W * 0.02]
        gap = (near[0][1] - near[0][0]) / W if near else 0.0
        return sx / W, gap, gap >= MIN_GAP_FRAC, depth
    # 그림자가 없는 평평한 스캔: 중앙(±1.5%)을 포함하는 글자 없는 띠만 제본선 후보.
    # 좁은 띠(≤12%)는 가운데, 넓은 띠는 쪽 안쪽 여백이 들쭉날쭉한 경우가 많아 중앙을 띠 안쪽으로 보정해 사용.
    mid = W * center
    cand = [r for r in runs if (r[1] - r[0]) >= W * MIN_GAP_FRAC and r[0] <= mid + 0.015 * W and r[1] >= mid - 0.015 * W]
    if not cand:
        return center, 0.0, False, depth
    r = max(cand, key=lambda r: r[1] - r[0])
    w = r[1] - r[0]
    x = (r[0] + r[1]) / 2 if w <= 0.12 * W else min(max(mid, r[0] + 0.1 * w), r[1] - 0.1 * w)
    return x / W, w / W, True, depth


def _gutter_from(g: np.ndarray, ink: np.ndarray) -> Gutter:
    H, W = g.shape
    top, bot = _text_rows(ink)                       # 위아래 가장자리(책상 등) 제외
    third = (bot - top) // 3
    center = 0.5        # 글자 분포로 책 중심을 추정하면 왼쪽 정렬 본문 때문에 쏠려서 쓰지 않는다. 치우친 책은 그림자·prior 가 처리
    # 전체 이미지 기준 그림자 위치(힌트): 그림자가 있으면 구간별 탐색을 그 근처로 제한
    lo, hi = int(W * BAND[0]), int(W * BAND[1])
    gb = _smooth(_bg_profile(g[top:bot]), int(W * 0.01))
    hx = lo + int(np.argmin(gb[lo:hi]))
    hint_ok = _valley_depth(gb, hx, W, lo, hi) > 0.03
    xs, gaps, founds, depths = [], [], [], []
    for k in range(3):
        a = top + k * third
        x, gap, ok, d = _find_in_rows(g, ink, a, a + third, center, hx / W, hint_ok)
        xs.append(x); gaps.append(gap); founds.append(ok); depths.append(d)
    med = float(np.median(xs))
    xs = [x if abs(x - med) <= 0.04 else med for x in xs]   # 한 구간만 크게 벗어난 값은 이상치로 보고 대체
    return Gutter(x=med, x_min=float(min(xs)), x_max=float(max(xs)),
                  gap=float(np.median(gaps)), found=sum(founds) >= 2,
                  shadow=float(np.median(depths)))      # 책등 전체를 따라 있는 그림자만: 한 구간에만 있는 어두운 그림은 중앙값에서 걸러진다


def find_gutter(img: Image.Image) -> Gutter:
    g = _gray_small(img)
    return _gutter_from(g, _drop_rules(_ink_mask(g)))


def _side_balance(mass: np.ndarray, x: float) -> float:
    xi = int(x * len(mass))
    left, right = float(mass[:xi].sum()), float(mass[xi:].sum())
    tot = left + right
    return min(left, right) / tot if tot > 0 else 0.0


def score_of(aspect: float, gap_e: float, shadow_e: float, side_e: float) -> float:
    """제본 증거 × 좌우 균형. 세로 이미지는 그림자를 필수 증거로 삼는다(두 단으로 짠 일반 쪽의 가운데 띠를 오인하지 않게)."""
    land = 1 - (1 - gap_e) * (1 - shadow_e)
    port = shadow_e * (0.6 + 0.4 * gap_e)
    w = _clamp((aspect - PORTRAIT_ASPECT) / (MIN_ASPECT - PORTRAIT_ASPECT))
    ev = w * land + (1 - w) * port
    return ev * (0.55 + 0.45 * side_e)


def analyze(img: Image.Image) -> Analysis:
    """프레임 하나의 펼침면 점수와 제본선. 결정(decision)은 decide() 에서 임계값·주변 프레임과 함께 내린다."""
    w, h = img.size
    g = _gray_small(img)
    ink = _drop_rules(_ink_mask(g))
    top, bot = _text_rows(ink)
    mass = ink[top:bot].mean(axis=0).astype(np.float32)
    gut = _gutter_from(g, ink)
    side = _side_balance(mass, gut.x)
    aspect = w / h
    gap_e = _clamp((gut.gap - 0.015) / 0.045) if gut.gap > 0 else 0.0
    shadow_e = _clamp(gut.shadow / 0.10)
    side_e = _clamp(side / 0.20)
    score = score_of(aspect, gap_e, shadow_e, side_e)
    if aspect >= MIN_ASPECT and gut.found and gut.gap >= MIN_GAP_FRAC:
        score = max(score, LEGACY_FLOOR)           # 이전 버전과 같은 판정(가로 이미지 + 띠)은 임계 기본값에서 반드시 유지
    third = (bot - top) // 3
    bright = np.stack([_smooth(_bg_profile(g[top + k * third: top + (k + 1) * third]), int(g.shape[1] * 0.01))
                       for k in range(3)]).astype(np.float32)
    return Analysis(aspect=aspect, gutter=gut, side=side, gap_e=gap_e, shadow_e=shadow_e, side_e=side_e,
                    score=float(score), mass=mass, bright=bright)


def _local_evidence(a: Analysis, x: float) -> tuple[float, float, float]:
    """x 위치(폭 비율)에서 (글자 밀도 비율, 그림자 깊이, 좌우 균형). 밀도 비율이 낮을수록 비어 있다."""
    W = len(a.mass)
    dens = _smooth(a.mass, int(W * 0.012))
    ref = float(np.percentile(dens, 75)) or 1e-6
    c, w = int(x * W), max(1, int(W * 0.02))
    blank = float(dens[max(0, c - w):c + w + 1].mean()) / ref
    lo, hi = max(0, int((x - 0.03) * W)), min(W, int((x + 0.03) * W) + 1)
    blo, bhi = int(W * BAND[0]), int(W * BAND[1])
    depth = float(np.median([max(_valley_depth(b, xi, W, blo, bhi) for xi in range(lo, hi)) for b in a.bright]))
    return blank, depth, _clamp(_side_balance(a.mass, x) / 0.20)


def decide(items: list[Analysis], mode: str = "auto", thr: float = DEFAULT_THR, use_prior: bool = True) -> dict:
    """프레임별 분리 여부(decision)와 사유(reason)를 채운다. 반환: 요약 통계."""
    for a in items:
        if mode == "off":
            a.decision, a.reason = False, "--spread off"
        elif mode == "on":
            a.decision, a.reason = True, "--spread on (모두 분리)"
        elif a.score >= thr:
            a.decision = True
            a.reason = f"점수 {a.score:.2f} ≥ 임계 {thr:.2f}"
        else:
            a.decision = False
            a.reason = f"점수 {a.score:.2f} < 임계 {thr:.2f}" + ("" if a.evidence >= 0.25 else " (제본 증거 없음)")
    stat = {"prior_x": None, "prior_n": 0, "promoted": 0}
    if mode == "auto" and use_prior:
        strong = [a for a in items if a.decision and a.score >= max(thr, STRONG)]
        if len(strong) >= 3 and len(strong) / len(items) >= 0.25:
            px = float(np.median([a.gutter.x for a in strong]))
            pmin = float(np.median([a.gutter.x_min for a in strong]))
            pmax = float(np.median([a.gutter.x_max for a in strong]))
            stat.update(prior_x=px, prior_n=len(strong))
            for a in items:
                if a.decision:
                    continue
                blank, depth, side_e = _local_evidence(a, px)
                if (blank <= 0.35 or depth >= 0.03) and side_e >= 0.5:
                    a.decision, a.prior = True, True
                    a.gutter = Gutter(px, pmin, pmax, a.gutter.gap, False, max(a.gutter.shadow, depth))
                    why = "띠" if blank <= 0.35 else "그림자"
                    a.reason = (f"점수 {a.score:.2f} < 임계 {thr:.2f} 이지만 확실한 {len(strong)}장의 제본선 위치"
                                f"(x={px:.3f})에서 {why} 확인 → 분리")
                    stat["promoted"] += 1
    return stat


def is_spread(img: Image.Image, mode: str = "auto", thr: float = DEFAULT_THR) -> tuple[bool, Gutter | None]:
    """mode: auto | on | off. auto 는 점수(제본 띠·그림자·좌우 균형) ≥ thr 이면 분리."""
    if mode == "off":
        return False, None
    if mode == "on":
        return True, find_gutter(img)
    a = analyze(img)
    return a.score >= thr, a.gutter


def split_image(img: Image.Image, gutter: Gutter, offset: float = 0.0, overlap: float = 0.015
                ) -> tuple[Image.Image, Image.Image]:
    """왼쪽/오른쪽 이미지. offset: 제본선 수동 보정(폭 비율, +는 오른쪽). overlap: 안쪽으로 겹칠 폭 비율."""
    w, h = img.size
    left_end = min(w, int((gutter.x_max + offset + overlap) * w))
    right_start = max(0, int((gutter.x_min + offset - overlap) * w))
    return img.crop((0, 0, left_end, h)), img.crop((right_start, 0, w, h))


CSV_COLS = ["frame", "source", "width", "height", "aspect", "decision", "score", "gap", "gap_e", "shadow", "shadow_e",
            "side", "gutter_x", "x_min", "x_max", "prior", "uncertain", "reason"]


def frame_row(i: int, name: str, size: tuple[int, int], a: Analysis | None, thr: float, mode: str) -> dict:
    w, h = size
    if a is None:
        return {"frame": i, "source": name, "width": w, "height": h, "aspect": round(w / h, 3), "decision": "single",
                "score": "", "gap": "", "gap_e": "", "shadow": "", "shadow_e": "", "side": "", "gutter_x": "",
                "x_min": "", "x_max": "", "prior": False, "uncertain": False, "reason": "--spread off"}
    g = a.gutter
    unc = a.prior or (mode == "auto" and abs(a.score - thr) <= UNCERTAIN_BAND)
    return {"frame": i, "source": name, "width": w, "height": h, "aspect": round(a.aspect, 3),
            "decision": "split" if a.decision else "single", "score": round(a.score, 3), "gap": round(g.gap, 3),
            "gap_e": round(a.gap_e, 2), "shadow": round(g.shadow, 3), "shadow_e": round(a.shadow_e, 2),
            "side": round(a.side, 3), "gutter_x": round(g.x, 3), "x_min": round(g.x_min, 3), "x_max": round(g.x_max, 3),
            "prior": a.prior, "uncertain": unc, "reason": a.reason}


def write_csv(path: Path, frames: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글이 깨지지 않게
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(frames)


def expand_spreads_ex(images: list[Path], out_dir: Path, mode: str = "auto", offset: float = 0.0,
                      overlap: float = 0.015, log=print, thr: float = DEFAULT_THR, verbose: bool = False
                      ) -> tuple[list[Path], list[dict], list[dict]]:
    """원본 이미지 목록 → 펼침면은 (왼, 오른) 두 장으로 분리한 페이지 목록. 좌→우 순서 유지.
    반환: (쪽 이미지, 쪽별 정보, 프레임별 판정 로그). 이미 만들어진 파일은 다시 만들지 않는다(재개용).
    판정 로그는 out_dir/spread_log.csv 로도 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes: list[tuple[int, int]] = []
    analyses: list[Analysis | None] = []
    for p in images:
        with Image.open(p) as im:
            sizes.append(im.size)
            analyses.append(None if mode == "off" else
                            analyze(im))
    live = [a for a in analyses if a is not None]
    stat = decide(live, mode, thr) if live else {"prior_x": None, "prior_n": 0, "promoted": 0}
    frames = [frame_row(i, p.name, sizes[i - 1], analyses[i - 1], thr, mode) for i, p in enumerate(images, start=1)]
    if verbose:
        for r in frames:
            tag = "분리" if r["decision"] == "split" else "단일"
            sc = f"점수 {r['score']:.2f}" if r["score"] != "" else "점수 -"
            log(f"  프레임 {r['frame']:>3} {r['source']} {r['width']}x{r['height']} {sc} → {tag}"
                f"{'*' if r['uncertain'] else ''} | {r['reason']}")
    pages: list[Path] = []
    infos: list[dict] = []
    n = 0
    for src_idx, p in enumerate(images, start=1):
        a = analyses[src_idx - 1]
        with Image.open(p) as im:
            im.load()
            extra = {"aspect": round(sizes[src_idx - 1][0] / sizes[src_idx - 1][1], 3)}
            if a is not None:
                extra.update(score=round(a.score, 3), evidence=round(a.evidence, 3))
            if a is None or not a.decision:
                n += 1
                dst = out_dir / f"p{n:04d}{p.suffix}"
                if not dst.exists():
                    im.save(dst, quality=95) if p.suffix.lower() in (".jpg", ".jpeg") else im.save(dst)
                pages.append(dst)
                infos.append({"page": n, "source": src_idx, "kind": "single", **extra})
                continue
            left, right = split_image(im, a.gutter, offset, overlap)
            for side, part in (("left", left), ("right", right)):
                n += 1
                dst = out_dir / f"p{n:04d}{p.suffix}"
                if not dst.exists():
                    part.save(dst, quality=95) if p.suffix.lower() in (".jpg", ".jpeg") else part.save(dst)
                pages.append(dst)
                infos.append({"page": n, "source": src_idx, "kind": side, "gutter": asdict(a.gutter), **extra})
            if not a.gutter.found and not a.prior:
                log(f"  [경고] 원본 {src_idx}번: 제본선(글자 없는 띠)을 못 찾아 그림자/중앙 기준으로 분할 — --gutter-offset 으로 보정하세요")
    (out_dir / "spreads.json").write_text(json.dumps(infos, ensure_ascii=False, indent=1), encoding="utf-8")
    write_csv(out_dir / "spread_log.csv", frames)
    if stat.get("prior_x") is not None:
        log(f"  책 전체 제본선 위치(확실한 {stat['prior_n']}장 기준): x={stat['prior_x']:.3f}, "
            f"증거가 약해 이 위치로 분리한 프레임 {stat['promoted']}장")
    return pages, infos, frames


def expand_spreads(images: list[Path], out_dir: Path, mode: str = "auto", offset: float = 0.0,
                   overlap: float = 0.015, log=print, thr: float = DEFAULT_THR) -> tuple[list[Path], list[dict]]:
    pages, infos, _ = expand_spreads_ex(images, out_dir, mode, offset, overlap, log, thr)
    return pages, infos
