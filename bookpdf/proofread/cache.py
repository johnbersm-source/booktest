"""호출 결과 캐시(JSON 줄 파일). 같은 요청을 다시 보내지 않아 호출 한도·비용을 아낀다. 책 본문이 들어갈 수 있으니 커밋 금지 폴더에만 둔다."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def key_of(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]


class ResponseCache:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self.data: dict[str, str] = {}
        if self.path and self.path.exists():
            for ln in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(ln)
                    self.data[r["k"]] = r["v"]
                except (ValueError, KeyError):
                    continue
        self.hits = 0

    def get(self, k: str) -> str | None:
        v = self.data.get(k)
        if v is not None:
            self.hits += 1
        return v

    def put(self, k: str, v: str) -> None:
        self.data[k] = v
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")
