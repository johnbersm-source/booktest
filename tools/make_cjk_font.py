"""Noto Sans CJK KR(OFL) → 책 PDF 용 부분집합 TrueType(glyf) 글꼴 생성.

reportlab(xhtml2pdf)은 CFF(OTF/TTC) 글꼴을 읽지 못하므로 필요한 글자만 뽑아 TrueType 윤곽으로 바꿉니다.
사용: python tools/make_cjk_font.py /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc bookpdf/fonts/NotoSansCJKkr-Regular.ttf
(fonttools 필요. 결과 .ttf 는 저장소에 번들되어 있으므로 일반 사용자는 실행할 필요 없음)
"""
from __future__ import annotations

import sys
from pathlib import Path

from fontTools import subset
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTCollection, TTFont, newTable

RANGES = [  # (시작, 끝) — 이 범위 중 글꼴에 실제 있는 글자만 담김
    (0x2100, 0x218F), (0x2190, 0x21FF), (0x2200, 0x22FF), (0x2460, 0x24FF), (0x2500, 0x25FF),
    (0x2600, 0x26FF), (0x2700, 0x27BF), (0x3000, 0x303F), (0x3040, 0x30FF), (0x3130, 0x318F),
    (0x3200, 0x33FF), (0xFE30, 0xFE4F), (0xFF00, 0xFFEF),
]


def ksx1001_hanja() -> set[int]:
    """KS X 1001 한자 4888자(EUC-KR 0xCA–0xFD 행)."""
    out = set()
    for hi in range(0xCA, 0xFE):
        for lo in range(0xA1, 0xFF):
            try:
                ch = bytes([hi, lo]).decode("euc_kr")
            except UnicodeDecodeError:
                continue
            if 0x4E00 <= ord(ch) <= 0x9FFF or 0xF900 <= ord(ch) <= 0xFAFF:
                out.add(ord(ch))
    return out


def to_truetype(font: TTFont) -> None:
    """CFF 윤곽 → 2차 곡선(glyf). 글리프 순서·메트릭은 유지."""
    order = font.getGlyphOrder()
    gs = font.getGlyphSet()
    glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, max_err=1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    maxp = font["maxp"]
    maxp.tableVersion = 0x00010000
    for a in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, a, 0)
    maxp.maxZones = 1
    maxp.maxPoints = maxp.maxContours = maxp.maxCompositePoints = maxp.maxCompositeContours = 0
    maxp.maxComponentDepth = 0
    font["head"].glyphDataFormat = 0
    font["post"].formatType = 2.0
    font["post"].extraNames, font["post"].mapping, font["post"].glyphOrder = [], {}, order
    font.sfntVersion = "\x00\x01\x00\x00"


def main(src: str, dst: str) -> None:
    ttc = TTCollection(src)
    kr = next(f for f in ttc.fonts if f["name"].getDebugName(1) == "Noto Sans CJK KR")
    cmap = kr.getBestCmap()
    want = {c for a, b in RANGES for c in range(a, b + 1) if c in cmap} | (ksx1001_hanja() & set(cmap))
    opts = subset.Options()
    opts.layout_features = ["kern", "vert", "vrt2"]
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    opts.glyph_names = False
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=sorted(want))
    sub.subset(kr)
    to_truetype(kr)
    for t in ("GSUB", "GPOS", "GDEF", "BASE", "vhea", "vmtx"):
        if t in kr:
            del kr[t]
    kr.save(dst)
    print(f"{dst}: {len(want)}자, {Path(dst).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
