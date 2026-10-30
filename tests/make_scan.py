"""tesseract 시험용 합성 '스캔 책' PDF(이미지만 있는 PDF). 제목(큰 글자)·본문·반복 머리글·쪽번호·차트 쪽 포함."""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookpdf.config import FONT_BOLD, FONT_REGULAR  # noqa: E402

W, H = 2480, 3508  # A4 300dpi, 본문 44px ≈ 10.5pt

PARAS = [
    "바다는 아침마다 다른 얼굴로 사람들을 맞이했다. 항구의 노인은 오래된 지도를 펼쳐 놓고 별빛이 닿는 방향을 천천히 읽어 내려갔다.",
    "시장 골목에는 새벽부터 불빛이 켜지고 종이 포장지에 싼 생선이 가지런히 놓였다. 상인들은 서로의 안부를 짧게 묻고 하루를 시작했다.",
    "기차는 다리를 건너 언덕을 지나갔고 창문 너머로 겨울 들판의 안개가 길게 흘러갔다. 승객들은 말없이 같은 풍경을 바라보았다.",
]


def _wrap(d: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split(" "):
        t = (cur + " " + word).strip()
        if d.textlength(t, font=font) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def page_image(n: int, kind: str, chapter: str, heading: str | None, printed: int) -> Image.Image:
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    body = ImageFont.truetype(str(FONT_REGULAR), 44)
    big = ImageFont.truetype(str(FONT_BOLD), 78)
    mid = ImageFont.truetype(str(FONT_BOLD), 58)
    small = ImageFont.truetype(str(FONT_REGULAR), 32)
    d.text((240, 140), "시험용 바다 이야기", font=small, fill="black")          # 반복 머리글
    d.text((W // 2 - 20, H - 180), str(printed), font=small, fill="black")       # 쪽번호
    y = 400
    if kind == "chart":
        d.text((240, y), "그림 3 분기별 매출", font=body, fill="black")
        for i, h in enumerate([400, 760, 600, 1120, 940]):
            x = 400 + i * 340
            d.rectangle([x, 2600 - h, x + 220, 2600], fill=(30 + i * 40, 90, 200 - i * 30))
        d.line([360, 2600, 2200, 2600], fill="black", width=8)
        d.line([360, 1200, 360, 2600], fill="black", width=8)
        return im
    if heading:
        d.text((240, y), heading, font=big if chapter == "big" else mid, fill="black")
        y += 170
    for p in PARAS * 3:
        for ln in _wrap(d, p, body, W - 480):
            d.text((240, y), ln, font=body, fill="black")
            y += 68
        y += 30
    return im


def build_scan_pdf(out: Path, spread: bool = False) -> list[str]:
    """6쪽: 1(제1장 제목+본문) 2(본문) 3(차트) 4(2.1 절 제목) 5(본문) 6(본문). 쪽번호 11~16. 반환: 기대 제목들."""
    specs = [("text", "big", "제1장 바다의 아침"), ("text", "", None), ("chart", "", None),
             ("text", "mid", "2.1 시장 골목"), ("text", "", None), ("text", "", None)]
    doc = fitz.open()
    for i, (kind, ch, hd) in enumerate(specs):
        im = page_image(i + 1, kind, ch, hd, 11 + i)
        tmp = out.parent / f"_scan_{i}.png"
        im.save(tmp)
        pg = doc.new_page(width=W * 72 / 300, height=H * 72 / 300)
        pg.insert_image(pg.rect, filename=str(tmp))
        tmp.unlink()
    doc.save(out)
    return ["제1장 바다의 아침", "2.1 시장 골목"]
