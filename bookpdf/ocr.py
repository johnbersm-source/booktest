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
6. 머리글/바닥글은 본문에서 제외하세요. 페이지 하단·상단에 인쇄된 쪽번호(아라비아 숫자)는 본문에 넣지 말고 page_number 에만 정수로 적으세요(없거나 읽을 수 없으면 null).
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
        "page_number": {"type": "integer", "nullable": True},
    },
    "required": ["markdown", "figures", "confidence"],
}


class QuotaExhausted(RuntimeError):
    """호출 한도 때문에 더 진행할 수 없음. kind: daily(일일, 몇 시간 기다려도 안 풀림) / minute(분당 한도가 계속 막음) / unknown."""

    def __init__(self, msg: str = "", kind: str = "daily"):
        super().__init__(msg)
        self.kind = kind


KIND_LABEL = {"daily": "일일 한도(RPD)", "minute": "분당 한도(RPM)", "unknown": "한도 종류 불명"}


def is_daily_quota_error(err: Exception) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("perday", "per day", "daily", "requestsperday"))


def classify_quota_error(err: Exception) -> str | None:
    """429/RESOURCE_EXHAUSTED 오류를 daily / minute / unknown 으로 구분. 한도 오류가 아니면 None."""
    msg = str(err).lower()
    if not (getattr(err, "code", None) == 429 or "resource_exhausted" in msg or "429" in msg):
        return None
    if is_daily_quota_error(err):
        return "daily"
    if any(k in msg for k in ("perminute", "per minute", "per_minute", "requestsperminute")):
        return "minute"
    return "unknown"


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
    pn = data.get("page_number")
    try:
        pn = int(pn) if pn is not None and str(pn).strip().lstrip("-").isdigit() else None
    except (TypeError, ValueError):
        pn = None
    if pn is not None and not 0 < pn < 10000:
        pn = None
    return {
        "markdown": str(data["markdown"]),
        "figures": figs,
        "confidence": max(0.0, min(1.0, conf)),
        "page_number": pn,
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
        last_kind: str | None = None
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
                kind = classify_quota_error(e)
                if kind == "daily":
                    raise QuotaExhausted(f"일일 한도(RPD) 도달: {str(e)[:200]}", "daily") from e
                delay = _retry_delay(e, attempt)
                if delay is None:
                    raise
                last, last_kind = e, kind
                what = {"minute": "분당 한도(RPM) 초과", "unknown": "429 (분당/일일 구분 불가)"}.get(kind or "", "일시 오류")
                log(f"  {what}, {delay:.0f}초 대기 후 재시도 {attempt + 1}/{max_attempts}: {str(e)[:160]}")
                time.sleep(delay)
        if last_kind:  # 429 가 끝까지 풀리지 않음: 다음 쪽으로 넘어가 계속 두드리지 말고 멈춘다
            hint = "GEMINI_RPM 을 낮춰 다시 실행" if last_kind == "minute" else "잠시 후(또는 내일) 다시 실행"
            raise QuotaExhausted(f"{KIND_LABEL[last_kind]}: 429 가 {max_attempts}회 연속 — {hint}. {str(last)[:160]}", last_kind)
        raise RuntimeError(f"OCR 실패(재시도 {max_attempts}회 소진): {last}")

    return ocr
