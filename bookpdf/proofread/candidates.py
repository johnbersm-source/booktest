"""M3 후보 생성: 책 안의 어휘(빈도)와 자모 편집거리로 수정 후보를 만든다.

OCR 오인식은 자모 모양이 비슷한 경우가 많아(ㅏ↔ㅓ, ㅗ↔ㅜ, ㄱ↔ㅋ …) 한글을 자모로 분해해 거리를 재고,
모양이 비슷한 자모끼리의 치환은 비용을 낮춘다(util.CONFUSABLE).
후보는 항상 '근거'(빈도·편집거리·출처)를 함께 기록한다. 책에 없는 단어를 지어내지 않는다 — 후보는 책 어휘·용어집·약어·두 번째 OCR·번호 규칙에서만 나온다.

탐색은 SymSpell 방식(삭제 변형 색인)으로 빠르게 후보를 좁힌 뒤 정확한 거리를 재서 거른다.
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

from . import util
from .normalize import Protect

MAX_INDEX_JAMO = 18        # 이보다 긴 자모열 단어는 색인하지 않는다(메모리 보호)


def _deletes(s: str, max_d: int) -> set[str]:
    out = {s}
    frontier = {s}
    for _ in range(max_d):
        nxt = set()
        for w in frontier:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1:])
        out |= nxt
        frontier = nxt
    return out


class _SymIndex:
    """문자열(자모열)을 키로 하는 삭제 변형 색인."""

    def __init__(self, max_d: int = 2):
        self.max_d = max_d
        self.table: dict[str, list[str]] = {}
        self.keys: dict[str, str] = {}          # 단어 → 색인 키(자모열 또는 소문자)

    def add(self, word: str, key: str) -> None:
        if len(key) > MAX_INDEX_JAMO or word in self.keys:
            return
        self.keys[word] = key
        for d in _deletes(key, self.max_d):
            self.table.setdefault(d, []).append(word)

    def near(self, key: str, max_d: int) -> set[str]:
        found: set[str] = set()
        for d in _deletes(key, min(max_d, self.max_d)):
            for w in self.table.get(d, ()):
                found.add(w)
        return found


class Vocab:
    """책 어휘: 한글 연속열(조사 포함)과 어간, 영문 단어의 빈도. 용어집·약어는 빈도와 무관하게 후보가 될 수 있다."""

    def __init__(self, protect: Protect | None = None, min_cand_freq: int = 2):
        self.protect = protect or Protect()
        self.min_cand_freq = min_cand_freq
        self.runs: Counter[str] = Counter()      # 한글 연속열(조사 붙은 채)
        self.stems: Counter[str] = Counter()     # 조사 뗀 어간(조사가 없으면 연속열 그대로)
        self.en: Counter[str] = Counter()        # 영문 단어(대소문자 구분)
        self.en_lower: dict[str, str] = {}       # 소문자 → 가장 흔한 표기
        self._ko_idx = _SymIndex(2)
        self._en_idx = _SymIndex(2)
        self._built = False

    # --- 구축
    def add_text(self, text: str) -> None:
        for line in text.split("\n"):
            for kind, w, _, _ in util.iter_words(line):
                if kind == "ko":
                    self.runs[w] += 1
                    stem, _j = util.split_josa(w)
                    self.stems[stem] += 1
                elif kind == "en" and len(w) >= 2:
                    self.en[w] += 1
        self._built = False

    def build(self) -> "Vocab":
        self._ko_idx = _SymIndex(2)
        self._en_idx = _SymIndex(2)
        for w, c in list(self.stems.items()) + list(self.runs.items()):
            if (c >= self.min_cand_freq or w in self.protect) and len(w) >= 2:
                self._ko_idx.add(w, util.to_jamo(w))
        for t in self.protect.terms:
            if util.has_hangul(t) and " " not in t and len(t) >= 2:
                self._ko_idx.add(t, util.to_jamo(t))
        lower: dict[str, tuple[int, str]] = {}
        for w, c in self.en.items():
            if lower.get(w.lower(), (0, ""))[0] < c:
                lower[w.lower()] = (c, w)
        self.en_lower = {k: v[1] for k, v in lower.items()}
        for w, c in self.en.items():
            if c >= self.min_cand_freq and len(w) >= 3:
                self._en_idx.add(w, w.lower())
        for a in self.protect.all_terms():
            if a.isascii() and len(a) >= 2 and any(ch.isalpha() for ch in a):
                self._en_idx.add(a, a.lower())
        self._built = True
        return self

    # --- 빈도
    def freq_ko(self, word: str) -> int:
        """한글 연속열의 알려진 정도: 연속열 자체 또는 어간 빈도 중 큰 값."""
        stem, _ = util.split_josa(word)
        return max(self.runs.get(word, 0), self.stems.get(word, 0), self.stems.get(stem, 0))

    def freq_en(self, word: str) -> int:
        f = self.en.get(word, 0)
        if f == 0 and word.lower() in self.en_lower:
            f = self.en.get(self.en_lower[word.lower()], 0)
        return f

    def is_known_en(self, word: str) -> bool:
        return word in self.protect

    # --- 후보
    def lookup_ko(self, word: str, max_wdist: float = 1.5, min_ratio: float = 2.0, limit: int = 5) -> list[dict]:
        """한글 연속열 word 와 자모 거리가 가까운, 더 흔한 책 어휘. 조사가 붙어 있으면 어간으로도 찾아 조사를 다시 붙인다."""
        assert self._built, "Vocab.build() 를 먼저 호출하세요"
        own = self.freq_ko(word)
        stem, josa = util.split_josa(word)
        out: dict[str, dict] = {}
        trials = [(word, "")]
        if josa:
            trials.append((stem, josa))
        for base, suffix in trials:
            if len(base) < 2:
                continue
            key = util.to_jamo(base)
            budget = 1 if len(base) <= 2 else 2
            for cand in self._ko_idx.near(key, budget):
                if cand == base:
                    continue
                dj = util.levenshtein(key, util.to_jamo(cand))
                if dj > budget:
                    continue
                wd = util.weighted_jamo_distance(base, cand)
                if wd > max_wdist:
                    continue
                cf = max(self.runs.get(cand, 0), self.stems.get(cand, 0))
                is_term = cand in self.protect
                if not is_term and (cf < self.min_cand_freq or cf < min_ratio * max(own, 1)):
                    continue
                text = cand + suffix
                if text == word:
                    continue
                rec = {"text": text, "kind": "ko", "source": "glossary" if is_term else "vocab",
                       "freq": cf, "own_freq": own, "dist": dj, "wdist": round(wd, 2)}
                prev = out.get(text)
                if prev is None or rec["wdist"] < prev["wdist"]:
                    out[text] = rec
        return rank(list(out.values()))[:limit]

    def lookup_en(self, word: str, max_dist: int | None = None, limit: int = 5) -> list[dict]:
        """영문(숫자 섞임 포함) 토큰과 문자 거리가 가까운 약어·용어·책 안의 흔한 영단어."""
        assert self._built
        low = word.lower()
        budget = max_dist if max_dist is not None else (1 if len(word) <= 4 else 2)
        out = []
        for cand in self._en_idx.near(low, budget):
            if cand == word:
                continue
            ck = cand.lower()
            d = util.levenshtein(low, ck)
            if d == 0 or d > budget:
                continue
            cf = self.en.get(cand, 0)
            term = cand in self.protect
            if not term and cf < self.min_cand_freq:
                continue
            out.append({"text": cand, "kind": "en", "source": "acronym" if cand in self.protect.acronyms else
                        ("glossary" if term else "vocab"), "freq": cf, "own_freq": self.freq_en(word),
                        "dist": d, "wdist": float(d)})
        return rank(out)[:limit]


def rank(cands: list[dict]) -> list[dict]:
    """가까울수록·흔할수록·용어집/약어 출처일수록 위. 점수 0~1 도 채운다."""
    for c in cands:
        base = 1.0 / (1.0 + c["wdist"])                     # 거리 0.4 → 0.71, 1 → 0.5, 2 → 0.33
        freq = min(1.0, math.log1p(c["freq"]) / math.log1p(20))
        bonus = 0.1 if c["source"] in ("glossary", "acronym") else 0.0
        c["score"] = round(min(1.0, 0.55 * base + 0.35 * freq + bonus), 3)
    return sorted(cands, key=lambda c: (-c["score"], c["wdist"], c["text"]))


def build_vocab(texts: list[str], protect: Protect | None = None, min_cand_freq: int = 2) -> Vocab:
    v = Vocab(protect, min_cand_freq)
    for t in texts:
        v.add_text(t)
    return v.build()


NOISE_CHARS = set("lI|1O0")


def strip_noise_candidates(vocab: Vocab, token: str) -> list[dict]:
    """'한l글'처럼 한글 사이에 낀 영문/숫자 한 글자 잡음을 지우면 책 어휘와 맞는지 본다."""
    out = []
    for i, ch in enumerate(token):
        if ch in NOISE_CHARS and 0 < i < len(token) - 1 and util.is_hangul_syllable(token[i - 1]) \
                and util.is_hangul_syllable(token[i + 1]):
            t2 = token[:i] + token[i + 1:]
            f = vocab.freq_ko(t2)
            if f >= vocab.min_cand_freq or t2 in vocab.protect:
                out.append({"text": t2, "kind": "ko", "source": "noise_removal", "freq": f, "own_freq": 0,
                            "dist": 1, "wdist": 1.0})
    return rank(out)


def gen_candidates(span: dict, vocab: Vocab, limit: int = 3) -> list[dict]:
    """span(detect 의 출력 한 건) → 후보 목록. 출처: vocab/glossary/acronym(책 어휘), alt_ocr(두 번째 OCR), sequence(번호 규칙), noise_removal.

    같은 text 가 여러 출처에서 나오면 합치고 sources 에 모두 기록한다(독립 근거가 많을수록 점수↑).
    """
    tok, kind = span["token"], span.get("kind", "ko")
    found: list[dict] = []
    if kind == "ko":
        found += vocab.lookup_ko(tok)
    elif kind in ("en", "mixed_en"):
        found += vocab.lookup_en(tok)
    elif kind == "mixed":
        found += strip_noise_candidates(vocab, tok)
        found += vocab.lookup_ko(tok)
    for sg in span.get("suggest", []):                      # detect 가 직접 제안한 값(번호 규칙 등)
        found.append({"text": sg["text"], "kind": kind, "source": sg["source"], "freq": 0, "own_freq": 0,
                      "dist": util.levenshtein(tok, sg["text"]), "wdist": float(util.levenshtein(tok, sg["text"]))})
    alt = span.get("alt")
    if alt and alt != tok:
        found.append({"text": alt, "kind": kind, "source": "alt_ocr",
                      "freq": vocab.freq_ko(alt) if util.has_hangul(alt) else vocab.freq_en(alt), "own_freq": 0,
                      "dist": util.levenshtein(tok, alt), "wdist": util.weighted_jamo_distance(tok, alt) if util.has_hangul(tok + alt) else float(util.levenshtein(tok, alt))})
    merged: dict[str, dict] = {}
    for c in found:
        m = merged.get(c["text"])
        if m is None:
            c = dict(c)
            c["sources"] = [c["source"]]
            merged[c["text"]] = c
        else:
            if c["source"] not in m["sources"]:
                m["sources"].append(c["source"])
            m["freq"] = max(m["freq"], c["freq"])
            m["wdist"] = min(m["wdist"], c["wdist"])
    out = list(merged.values())
    for c in out:
        c["wdist"] = round(c["wdist"], 2)
    ranked = rank(out)
    for c in ranked:                                         # 독립 출처가 둘 이상이면 가산
        if len(c["sources"]) >= 2:
            c["score"] = round(min(1.0, c["score"] + 0.15 * (len(c["sources"]) - 1)), 3)
    ranked.sort(key=lambda c: (-c["score"], c["wdist"], c["text"]))
    return ranked[:limit]
