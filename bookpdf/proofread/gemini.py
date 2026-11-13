"""교정용 Gemini 호출기. 재시도·429 처리·호출 간격은 bookpdf/ocr.py 의 로직을 그대로 재사용한다(수정하지 않음).

호출기 인터페이스(가짜 클라이언트도 이것만 지키면 됨):
    caller(prompt: str, *, image: bytes | None = None, mime: str = "image/png", schema: dict | None = None) -> str
    caller.calls  — 지금까지의 호출 횟수(재시도 포함)

모델: 환경변수 GEMINI_MODEL, 없으면 gemini-3.5-flash-lite (이 이름이 현재 실제로 존재하는지는 이 세션에서 확인하지 못함 → 막히면 .env 에서 바꾸세요).
API 키는 .env/환경변수로만 받는다. 이 모듈은 키를 로그에 남기지 않는다.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

from ..ocr import KIND_LABEL, QuotaExhausted, RateLimiter, _retry_delay, classify_quota_error

DEFAULT_MODEL = "gemini-3.5-flash-lite"


@dataclass
class GeminiSettings:
    api_key: str
    model: str
    rpm: float
    rpd: int


def load_settings() -> GeminiSettings:
    from dotenv import load_dotenv
    load_dotenv()
    return GeminiSettings(
        api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
        model=os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL,
        rpm=float(os.environ.get("GEMINI_RPM", "10") or 10),
        rpd=int(os.environ.get("GEMINI_RPD", "0") or 0),
    )


class GeminiCaller:
    def __init__(self, api_key: str, model: str, rpm: float, max_attempts: int = 6, max_calls: int = 0,
                 log: Callable[[str], None] = print, sleep=time.sleep, client=None, clock=time.monotonic):
        if client is None:
            from google import genai
            client = genai.Client(api_key=api_key)
        self.client, self.model, self.max_attempts, self.max_calls = client, model, max_attempts, max_calls
        self.log, self._sleep = log, sleep
        self.limiter = RateLimiter(rpm, clock=clock, sleep=sleep)
        self.calls = 0

    def __call__(self, prompt: str, *, image: bytes | None = None, mime: str = "image/png",
                 schema: dict | None = None) -> str:
        from google.genai import types
        contents: list = []
        if image is not None:
            contents.append(types.Part.from_bytes(data=image, mime_type=mime))
        contents.append(prompt)
        cfg = {"temperature": 0, "response_mime_type": "application/json"}
        if schema is not None:
            cfg["response_schema"] = schema
        last: Exception | None = None
        last_kind: str | None = None
        for attempt in range(self.max_attempts):
            if self.max_calls and self.calls >= self.max_calls:
                raise QuotaExhausted(f"이번 실행의 Gemini 호출 상한({self.max_calls}회)에 도달", "daily")
            self.limiter.wait()
            self.calls += 1
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=contents, config=types.GenerateContentConfig(**cfg))
                return resp.text or ""
            except Exception as e:  # noqa: BLE001
                kind = classify_quota_error(e)
                if kind == "daily":
                    raise QuotaExhausted(f"일일 한도(RPD) 도달: {str(e)[:200]}", "daily") from e
                delay = _retry_delay(e, attempt)
                if delay is None:
                    raise
                last, last_kind = e, kind
                what = {"minute": "분당 한도(RPM) 초과", "unknown": "429 (분당/일일 구분 불가)"}.get(kind or "", "일시 오류")
                self.log(f"  {what}, {delay:.0f}초 대기 후 재시도 {attempt + 1}/{self.max_attempts}: {str(e)[:160]}")
                self._sleep(delay)
        if last_kind:
            hint = "GEMINI_RPM 을 낮춰 다시 실행" if last_kind == "minute" else "잠시 후(또는 내일) 다시 실행"
            raise QuotaExhausted(f"{KIND_LABEL[last_kind]}: 429 가 {self.max_attempts}회 연속 — {hint}. {str(last)[:160]}", last_kind)
        raise RuntimeError(f"Gemini 호출 실패(재시도 {self.max_attempts}회 소진): {last}")


def make_caller(s: GeminiSettings, max_calls: int = 0, log: Callable[[str], None] = print) -> GeminiCaller:
    return GeminiCaller(s.api_key, s.model, s.rpm, max_calls=max_calls or s.rpd, log=log)


SEND_NOTICE = ("※ 의심 구간의 앞뒤 문맥(글자)과 쪽 이미지 조각이 Google Gemini API 로 전송됩니다. "
               "보내고 싶지 않으면 --no-llm --no-verify 로 실행하세요.")
