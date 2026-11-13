"""교정 파이프라인 테스트용 합성(가짜) 한국어 샘플. 실제 책 본문은 쓰지 않는다.

정답 문장 → 의도적으로 OCR 오류를 넣은 쪽을 만들고, work/pages/pNNNN.json 구조로 저장한다.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).resolve().parents[1] / "bookpdf" / "fonts" / "NanumGothic-Regular.ttf"

# 합성 정답(가짜 투자 교양서 문장). 서로 다른 쪽에 같은 용어가 반복되도록 구성.
GOLD = {
    1: ["# 제1장 이동평균선의 이해",
        "",
        "이동평균선은 일정 기간의 주가 평균을 이은 선이다. 이동평균선이 상승하면 추세가 살아 있다고 본다.",
        "2021년 코스피 지수는 3,000선을 넘어섰고 투자자들은 이동평균선과 거래량을 자주 확인했다."],
    2: ["MACD와 RSI는 대표적인 보조지표이다. 보조지표는 추세의 강도를 보여 준다.",
        "",
        "그림 1-1 이동평균선과 주가",
        "투자자는 이동평균선과 거래량을 함께 본다. 거래량이 늘면 추세의 신뢰도가 커진다."],
    3: ["보조지표만으로 매매하면 위험하다. 이동평균선과 거래량을 함께 확인해야 한다.",
        "",
        "그림 1-2 거래량의 변화",
        "2022년에는 금리 인상으로 변동성이 커졌다. 추세의 강도는 약해졌다."],
    4: ["| 구분 | 비중(%) |",
        "|---|---|",
        "| 주식 | 40 |",
        "| 채권 | 35 |",
        "| 현금 | 25 |",
        "| 합계 | 100 |",
        "",
        "이동평균선과 거래량은 투자의 기본 도구이다. 추세가 살아 있는지 보조지표로 확인한다."],
    5: ["추세의 방향을 먼저 판단한다. 이동평균선이 우상향하면 추세가 살아 있다고 본다.",
        "",
        "그림 1-3 보조지표 비교",
        "거래량이 줄면서 주가가 오르면 신뢰도가 낮다. 신뢰도는 거래량으로 확인한다."],
}


def blocks_of(md: str, size=(400, 300)) -> list[dict]:
    """tesseract 결과의 blocks 모양을 흉내: 빈 줄로 나뉜 문단 → 블록(본문 구역, 줄 높이 20)."""
    out, y = [], 20
    for para in [p for p in md.split("\n\n") if p.strip()]:
        n = para.count("\n") + 1
        out.append({"text": para, "zone": "body", "y0": y, "y1": y + 20 * n, "x0": 20, "x1": size[0] - 20, "lh": 20.0,
                    "lines": n, "chars": len(para), "conf": 90.0})
        y += 20 * n + 10
    return out


def make_work(tmp: Path, pages: dict[int, list[str]] | None = None, engine: str = "tesseract",
              size=(1000, 400), blocks: bool = False) -> Path:
    """work/pages/pNNNN.json 과 가짜 쪽 이미지를 만든다. 반환: work 경로."""
    work = Path(tmp) / "work"
    (work / "pages").mkdir(parents=True, exist_ok=True)
    (work / "page_images").mkdir(parents=True, exist_ok=True)
    for n, lines in (pages or GOLD).items():
        img = work / "page_images" / f"p{n:04d}.png"
        im = Image.new("RGB", size, "white")
        d = ImageDraw.Draw(im)
        d.text((10, 2), f"page {n}", fill="black")
        if blocks:                          # 실제 글자를 그려 줄마다 잘라낸 이미지가 서로 다르게 한다(번들 NanumGothic 사용)
            font = ImageFont.truetype(str(FONT), 14)
            for b in blocks_of("\n".join(lines), size):
                for k, ln in enumerate(b["text"].split("\n")):
                    d.text((b["x0"], b["y0"] + k * 20), ln, fill="black", font=font)
        im.save(img)
        md = "\n".join(lines)
        rec = {"page": n, "markdown": md, "figures": [], "confidence": 0.8, "engine": engine,
               "image": str(img), "page_number": n}
        if blocks:
            rec["blocks"] = blocks_of(md, size)
            rec["size"] = list(size)
            rec["ink"] = 0.02
            rec["chars"] = len(md)
        (work / "pages" / f"p{n:04d}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        (work / "pages" / f"p{n:04d}.md").write_text(md, encoding="utf-8")
    return work


def noisy(pages: dict[int, list[str]], subs: dict[int, list[tuple[str, str]]]) -> dict[int, list[str]]:
    """정답 쪽에 (원문, 오류) 치환을 적용한 OCR 쪽을 만든다. 치환 대상이 없으면 AssertionError(테스트 설계 오류 방지)."""
    out = {}
    for n, lines in pages.items():
        text = "\n".join(lines)
        for old, new in subs.get(n, []):
            assert old in text, f"p{n}: '{old}' 없음"
            text = text.replace(old, new, 1)
        out[n] = text.split("\n")
    return out


# 대표 오류 세트: 자모 혼동(살→삼, 량→랑, 도→드), 영문 약어 훼손(RSI→R5I), 숫자 혼동(2022→2O22)
SUBS = {
    1: [("추세가 살아", "추세가 삼아")],
    2: [("RSI는", "R5I는")],
    3: [("거래량을 함께", "거래랑을 함께"), ("2022년에는", "2O22년에는")],
    5: [("신뢰도가 낮다", "신뢰드가 낮다")],
}
# 위 SUBS 가 의도한 정답(테스트에서 검증): 오류 토큰 → 정답 토큰
TRUTH = {"삼아": "살아", "R5I": "RSI", "거래랑을": "거래량을", "신뢰드가": "신뢰도가", "2O22": "2022"}
