"""--pages 범위 문자열 해석: '114-215', '5', '1-3,10-12' → 정렬된 1부터의 쪽 번호 목록."""
from __future__ import annotations

import re


def parse_pages(spec: str, total: int | None = None) -> list[int]:
    """spec 을 쪽 번호 목록으로. 형식 오류/범위 초과는 ValueError (한국어 메시지)."""
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("--pages 값이 비어 있습니다. 예: 114-215 또는 1-3,10")
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if not m:
            raise ValueError(f"--pages 형식 오류: '{part}' (예: 114-215, 5, 1-3,10-12)")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        if a < 1 or b < a:
            raise ValueError(f"--pages 범위 오류: '{part}' (1 이상, 시작 ≤ 끝)")
        if total is not None and b > total:
            raise ValueError(f"--pages '{part}' 가 전체 {total}쪽을 넘습니다.")
        pages.update(range(a, b + 1))
    return sorted(pages)
