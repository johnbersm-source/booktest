"""Gemini 기반 페이지 OCR: 이미지 → Markdown(+표) + 그림/도표 좌표(box_2d)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Callable

PROMPT = """당신은 책 스캔 페이지를 디지털 텍스트로 옮기는 OCR 전문가입니다. 이 이미지는 책의 한 페이지입니다.
규칙:
1. 보이는 글자만 옮기고, 보이지 않는 내용을 지어내지 마세요. 읽을 수 없는 부분은 [판독불가] 로 표시.
2. 한국어 위주입니다. 원문 그대로(맞춤법 교정 금지) 옮기세요.
3. 제목은 Markdown 제목(#, ##, ###), 목록은 -, 1. 으로 표현하세요.
4. 표는 반드시 Markdown 표(| 열 | 열 |, 구분선 |---|---|)로 옮기세요.
5. 사진·그림·도표·그래프(글자만으로 옮길 수 없는 영역)는 figures 에 box_2d 로 등록하고,
   본문의 해당 위치에 {{FIG:n}} (n은 figures 순번, 1부터) 마커를 넣으세요. 표는 그림이 아니라 Markdown 표로 옮기세요.
6. 쪽번호와 머리글/바닥글은 본문에서 제외하세요.
7. 문단은 빈 줄로 구분하고, 같은 문단 안의 줄바꿈은 그대로 두세요.
8. confidence 는 이 페이지 전사의 신뢰도(0~1)를 정직하게 적으세요(흐림·잘림·손글씨면 낮게).
box_2d 는 [ymin, xmin, ymax, xmax], 이미지 전체를 0~1000으로 정규화한 정수입니다."""

SCHEMA = {
    "type": "object",
    "properties": {
        "markdown": {"type": "string"},
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {"type": "array", "items": {"type": "integer"}},
                    "caption": {"type": "string"},
                },
                "required": ["box_2d"],
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["markdown", "figures", "confidence"],
}


class RateLimiter:
    """무료 티어 RPM 한도를 넘지 않도록 호출 간격을 둔다."""

    def __init__(self, rpm: float, clock=time.monotonic, sleep=time.sleep):
        self.min_gap = 60.0 / rpm if rpm > 0 else 0.0
        self._last = None
        self._clock, self._sleep = clock, sleep

    def wait(self) -> None:
        if self._last is not None:
            gap = self._clock() - self._last
            if gap < self.min_gap:
                self._sleep(self.min_gap - gap)
        self._last = self._clock()


def _retry_delay(err: Exception, attempt: int) -> float | None:
    """재시도 가능한 오류면 대기 초, 아니면 None."""
    code = getattr(err, "code", None)
    msg = str(err)
    transient = code in (429, 500, 502, 503, 504) or any(
        k in msg for k in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE", "timed out", "Timeout")
    )
    if not transient:
        return None
    m = re.search(r"retry(?:Delay)?\D{0,12}(\d+(?:\.\d+)?)\s*s", msg, re.I)
    if m:
        return float(m.group(1)) + 1.0
    return min(60.0, 5.0 * (2 ** attempt))


def parse_response(text: str) -> dict:
    """모델 응답(JSON 문자열) → 검증된 dict. 형식이 틀리면 ValueError."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    data = json.loads(t)
    if not isinstance(data, dict) or "markdown" not in data:
        raise ValueError("응답에 markdown 필드가 없습니다.")
    figs = []
    for f in data.get("figures") or []:
        box = f.get("box_2d")
        if isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
            ymin, xmin, ymax, xmax = (max(0, min(1000, int(v))) for v in box)
            if ymax > ymin and xmax > xmin:
                figs.append({"box_2d": [ymin, xmin, ymax, xmax], "caption": str(f.get("caption") or "")})
    try:
        conf = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    return {
        "markdown": str(data["markdown"]),
        "figures": figs,
        "confidence": max(0.0, min(1.0, conf)),
        "engine": "gemini",
    }


def make_gemini_ocr(api_key: str, model: str, rpm: float, max_attempts: int = 6,
                    log: Callable[[str], None] = print) -> Callable[[Path], dict]:
    """페이지 이미지 경로 → 결과 dict 를 돌려주는 함수를 만든다."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    limiter = RateLimiter(rpm)

    def ocr(image_path: Path) -> dict:
        data = image_path.read_bytes()
        mime = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        last: Exception | None = None
        for attempt in range(max_attempts):
            limiter.wait()
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=[types.Part.from_bytes(data=data, mime_type=mime), PROMPT],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SCHEMA,
                        temperature=0,
                    ),
                )
                return parse_response(resp.text or "")
            except (ValueError, json.JSONDecodeError) as e:  # 형식 오류: 같은 요청을 다시 시도
                last = e
                log(f"  응답 형식 오류, 재시도 {attempt + 1}/{max_attempts}: {e}")
                time.sleep(2)
            except Exception as e:  # noqa: BLE001
                delay = _retry_delay(e, attempt)
                if delay is None:
                    raise
                last = e
                log(f"  호출 제한/일시 오류, {delay:.0f}초 대기 후 재시도 {attempt + 1}/{max_attempts}: {str(e)[:160]}")
                time.sleep(delay)
        raise RuntimeError(f"OCR 실패(재시도 {max_attempts}회 소진): {last}")

    return ocr
