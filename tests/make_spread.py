"""합성 펼침면 생성: 두 쪽 + 제본 그림자 + (선택) 기울기/곡률. 쪽마다 색 표식을 가장자리 근처에 둬서 잘림을 센다."""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from bookpdf.config import FONT_REGULAR

VOCAB = "바다 하늘 구름 나무 사랑 시간 마음 기억 도시 여행 노래 겨울 아침 강물 바람 언덕 시장 편지 학교 정원".split()


def page_img(seed: int, w: int = 700, h: int = 1000, marks: tuple = (200, 0, 0)) -> Image.Image:
    rnd = random.Random(seed)
    im = Image.new("RGB", (w, h), (250, 248, 240))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(str(FONT_REGULAR), 26)
    y = 90
    while y < h - 120:
        d.text((70, y), " ".join(rnd.sample(VOCAB, 7)), fill=(20, 20, 20), font=f)
        y += 44
    # 표식: 안쪽(제본선 쪽) 가장자리 근처 위/아래 — 잘림 검사용 (색은 쪽마다 다름)
    return im, marks


def make_spread(seed: int = 1, tilt_deg: float = 0.0, gutter_shadow: bool = True, w: int = 700, h: int = 1000,
                inner_margin: int = 70, bow_px: int = 0, marker_y: tuple | None = None) -> tuple[Image.Image, dict]:
    """왼쪽 쪽: 빨강 표식(오른쪽=안쪽 끝 근처), 오른쪽 쪽: 파랑 표식(왼쪽=안쪽 끝 근처).
    반환: (이미지, {'red': 총 빨강 픽셀수, 'blue': ...})"""
    L, _ = page_img(seed, w, h)
    R, _ = page_img(seed + 100, w, h)
    for im, color, inner_right in ((L, (220, 0, 0), True), (R, (0, 0, 220), False)):
        d = ImageDraw.Draw(im)
        x1 = w - inner_margin - 40 if inner_right else inner_margin
        for y in (marker_y or (40, h - 70)):    # 위/아래 모서리 근처(기울면 제본선 쪽이 가장 먼저 잘리는 곳)
            d.rectangle([x1, y, x1 + 40, y + 28], fill=color)
    # 쪽 안쪽 여백(제본 쪽)을 margin_gap 만큼 비워 글자 없는 띠를 만든다: 글자 오른쪽/왼쪽 70px 여백 이미 있음
    spread = Image.new("RGB", (2 * w, h), (250, 248, 240))
    spread.paste(L, (0, 0))
    spread.paste(R, (w, 0))
    arr = np.asarray(spread).astype(np.float32)
    if gutter_shadow:
        xs = np.arange(2 * w)
        shade = 1 - 0.28 * np.exp(-((xs - w) ** 2) / (2 * 22 ** 2))
        arr = arr * shade[None, :, None]
    spread = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    if bow_px:  # 책등이 활처럼 휜 효과: 행마다 가로로 sin 만큼 이동
        a = np.asarray(spread)
        out = np.empty_like(a)
        for y in range(h):
            s = int(round(bow_px * np.sin(np.pi * y / h)))
            out[y] = np.roll(a[y], s, axis=0)
        spread = Image.fromarray(out)
    if tilt_deg:
        spread = spread.rotate(tilt_deg, resample=Image.BICUBIC, expand=False, fillcolor=(250, 248, 240))
    spread = spread.filter(ImageFilter.GaussianBlur(0.6))
    return spread, {}


def count_color(img: Image.Image, color: str) -> int:
    a = np.asarray(img.convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (r > 120) & (g < 90) & (b < 90) if color == "red" else (b > 120) & (r < 90) & (g < 90)
    return int(m.sum())


DESK = (96, 78, 62)


def to_portrait_frame(spread: Image.Image, fw: int = 1080, fh: int = 1920, cx: float = 0.5, scale: float = 0.96,
                      cy: float = 0.5) -> Image.Image:
    """펼침면을 책상 위에 놓인 모양으로 세로 프레임(아이폰 세로 영상 프레임)에 넣는다. cx: 책 중심의 가로 위치."""
    s = fw * scale / spread.width
    sp = spread.resize((int(spread.width * s), int(spread.height * s)), Image.LANCZOS)
    fr = Image.new("RGB", (fw, fh), DESK)
    fr.paste(sp, (int(fw * cx - sp.width / 2), int(fh * cy - sp.height / 2)))
    return fr


def make_chart_spread(seed: int = 1, shadow: bool = True, w: int = 700, h: int = 1000, shadow_amp: float = 0.28) -> Image.Image:
    """제본선을 가로지르는 큰 차트(막대+선) 쪽. 가운데에 글자 없는 띠가 없어 띠 기반 판정이 실패하는 경우."""
    base, _ = make_spread(seed=seed, gutter_shadow=False, w=w, h=h)
    d = ImageDraw.Draw(base)
    d.rectangle([60, 200, 2 * w - 60, h - 200], fill=(250, 248, 240))
    rnd = random.Random(seed)
    n = 14
    bw = (2 * w - 160) // n
    for i in range(n):
        top = h - 260 - rnd.randint(80, 480)
        d.rectangle([80 + i * bw, top, 80 + i * bw + bw - 10, h - 260], fill=(60, 90, 170))
    pts = [(80 + i * bw + bw // 2, h - 260 - rnd.randint(150, 500)) for i in range(n)]
    d.line(pts, fill=(200, 60, 40), width=6)
    d.rectangle([60, 200, 2 * w - 60, h - 200], outline=(30, 30, 30), width=4)
    arr = np.asarray(base).astype(np.float32)
    if shadow:
        xs = np.arange(2 * w)
        arr = arr * (1 - shadow_amp * np.exp(-((xs - w) ** 2) / (2 * 22 ** 2)))[None, :, None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))
