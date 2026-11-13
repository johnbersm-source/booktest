"""교정 파이프라인 테스트용 합성(가짜) 한국어 샘플. 실제 책 본문은 쓰지 않는다.

정답 문장 → 의도적으로 OCR 오류를 넣은 쪽을 만들고, work/pages/pNNNN.json 구조로 저장한다.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

# 합성 정답(가짜 투자 교양서 문장). 서로 다른 쪽에 같은 용어가 반복되도록 구성.
GOLD = {
    1: ["# 제1장 이동평균선의 이해",
        "",
        "이동평균선은 일정 기간의 주가 평균을 이은 선이다. 이동평균선이 상승하면 추세가 살아 있다고 본다.",
        "2021년 코스피 지수는 3,000선을 넘어섰고 투자자들은 이동평균선을 자주 확인했다."],
    2: ["MACD와 RSI는 대표적인 보조지표이다. 보조지표는 추세의 강도를 보여 준다.",
        "",
        "그림 1-1 이동평균선과 주가",
        "투자자는 이동평균선과 거래량을 함께 본다. 거래량이 늘면 추세의 신뢰가 커진다."],
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
        "이동평균선과 거래량은 투자의 기본 도구이다. 보조지표는 참고로 쓴다."],
    5: ["추세의 방향을 먼저 판단한다. 이동평균선이 우상향하면 매수 관점을 유지한다.",
        "",
        "그림 1-3 보조지표 비교",
        "거래량이 줄면서 주가가 오르면 신뢰도가 낮다. 이동평균선을 함께 본다."],
}


def make_work(tmp: Path, pages: dict[int, list[str]] | None = None, engine: str = "tesseract",
              size=(400, 300)) -> Path:
    """work/pages/pNNNN.json 과 가짜 쪽 이미지를 만든다. 반환: work 경로."""
    work = Path(tmp) / "work"
    (work / "pages").mkdir(parents=True, exist_ok=True)
    (work / "page_images").mkdir(parents=True, exist_ok=True)
    for n, lines in (pages or GOLD).items():
        img = work / "page_images" / f"p{n:04d}.png"
        im = Image.new("RGB", size, "white")
        d = ImageDraw.Draw(im)
        d.text((10, 10), f"page {n}", fill="black")
        im.save(img)
        md = "\n".join(lines)
        rec = {"page": n, "markdown": md, "figures": [], "confidence": 0.8, "engine": engine,
               "image": str(img), "page_number": n}
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


# 대표 오류 세트: 한글 자모 혼동, 영문 약어 훼손, 숫자 혼동, 음절 사이 공백
SUBS = {
    1: [("이동평균선이 상승하면", "이동평균선이 상승하면"), ("추세가 살아", "추세가 삼아"), ("3,000선", "3,O00선")],
    2: [("보조지표는 추세", "보조지표는 추세"), ("RSI는", "R5I는"), ("신뢰가", "신뢰가")],
    3: [("거래량을 함께", "거래랑을 함께"), ("2022년에는", "2O22년에는")],
    4: [("거래량은 투자", "거래량은 투자")],
    5: [("우상향하면", "우상향하먼")],
}
