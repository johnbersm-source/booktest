import sys

if len(sys.argv) > 1 and sys.argv[1] == "proofread":      # 하위 명령: OCR 결과 교정 (bookpdf/proofread/)
    from .proofread.cli import main as _proofread_main
    raise SystemExit(_proofread_main(sys.argv[2:]))

from .cli import main

raise SystemExit(main())
