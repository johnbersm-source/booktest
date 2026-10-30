"""테스트용 샘플 책 PDF 생성 (한국어, 제목/표/그림/반복 머리글·쪽번호/쪽 경계 문단)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bookpdf import render  # noqa: E402


VOCAB = ("바다 하늘 구름 나무 사랑 시간 마음 기억 도시 여행 노래 겨울 아침 강물 바람 언덕 시장 편지 학교 정원 "
         "기차 다리 불빛 이야기 계절 거울 우산 약속 침묵 파도 종이 연필 창문 골목 새벽 그림자 호수 안개 노을 별빛").split()


def make_chart(path: Path) -> None:
    im = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 639, 359], outline="black", width=3)
    for i, h in enumerate([80, 150, 120, 230, 190]):
        x = 60 + i * 110
        d.rectangle([x, 330 - h, x + 70, 330], fill=(40 + i * 30, 90, 200 - i * 25))
    im.save(path)


def build(out_pdf: Path) -> None:
    work = out_pdf.parent
    img = work / "chart.png"
    make_chart(img)
    pages = []
    for p in range(1, 7):
        rnd = random.Random(p)
        sent = "".join(
            f"{p}장 {k}번째: " + " ".join(rnd.sample(VOCAB, 9)) + f" English word{p}{k} 숫자 {rnd.randint(100, 99999)}. "
            for k in range(1, 3))
        body = f"<h2>{p}장. 시험 단원 {p}</h2>"
        body += "<p>" + sent * 3 + "</p>"
        if p == 2:
            body += ('<table border="1" cellpadding="4"><tr><th>항목</th><th>값</th><th>비고</th></tr>'
                     '<tr><td>사과</td><td>1,200</td><td>국내산</td></tr>'
                     '<tr><td>배</td><td>3,400</td><td>수입</td></tr>'
                     '<tr><td>포도</td><td>2,800</td><td>국내산</td></tr></table>')
        if p == 3:
            body += f'<p><img src="{img.as_posix()}" width="300" height="169"/></p><p>그림 아래 설명 문장입니다.</p>'
        if p == 4:  # 문장이 다음 쪽으로 이어짐
            body += "<p>" + sent * 2 + "이 문장은 쪽 경계에서 끊기고 이어서</p>"
        if p == 5:
            body = "<p>다음 쪽에서 계속되는 나머지 문장입니다.</p>" + body
        pages.append(
            f'<div style="font-size:8pt;color:#666">샘플 책 · 머리글</div>{body}'
            f'<div style="font-size:8pt;text-align:center">- {p} -</div>' + ('<pdf:nextpage />' if p < 6 else ""))
    html = f"""<html><head><meta charset="utf-8"/><style>
@font-face {{ font-family: NanumGothic; src: url("{render.FONT_REGULAR.resolve().as_posix()}"); }}
@page {{ size: A4; margin: 2cm; }} body {{ font-family: NanumGothic; font-size: 11pt; line-height: 150%; }} h2 {{ font-size: 17pt; }}
</style></head><body>{''.join(pages)}</body></html>"""
    render.build_pdf(html, out_pdf, allowed_dirs=[work])


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("sample.pdf"))
