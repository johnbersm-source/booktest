"""환경설정. API 키는 .env / 환경변수로만 받는다."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
FONT_DIR = ROOT / "fonts"
FONT_REGULAR = FONT_DIR / "NanumGothic-Regular.ttf"
FONT_BOLD = FONT_DIR / "NanumGothic-Bold.ttf"


@dataclass
class Settings:
    api_key: str
    model: str
    rpm: float
    rpd: int = 0


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
        model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        rpm=float(os.environ.get("GEMINI_RPM", "10")),
        rpd=int(os.environ.get("GEMINI_RPD", "0") or 0),
    )
