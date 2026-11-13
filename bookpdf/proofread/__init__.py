"""OCR 결과 교정 파이프라인(보수적: 근거 있는 수정만 적용, 나머지는 보류 → 검수 리포트).

단계: normalize(M1) → detect(M2) → candidates(M3) → llm_edit(M4) → verify(M5) → report(M6) → evaluate(M7) → cli(M8)
기존 bookpdf 모듈은 읽기 전용으로만 재사용한다(ocr.py 의 Gemini 재시도·429 처리 등).
"""
