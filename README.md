# booktest — PDF(·로컬 전용 .mov)에서 글자를 뽑아 '책 PDF' 만들기

입력 PDF(또는 가정용 PC에서만: 아이폰 `.mov` 영상)를 페이지 이미지로 만들고, **Gemini**로 글자·표·그림 위치를 읽어
`output/책.pdf`(표지·목차·본문)를 만듭니다. 정확도가 완벽하지 않아도 책 PDF가 나오게 하고, 의심스러운 페이지는 리포트로 알려 줍니다.

## 확정된 설계 결정 (되돌리지 말 것)
- PDF 생성은 **xhtml2pdf + NanumGothic**(`bookpdf/fonts/`에 번들, OFL 라이선스). WeasyPrint는 쓰지 않음(Windows GTK 문제).
- OCR은 Claude API가 아니라 **Gemini**(무료/구독 티어). 키는 `.env`로만 받고 커밋하지 않음.
- 영상(.mov) 처리는 `bookpdf/local/` 에 분리된 **로컬 전용** 모듈. 영상 파일은 `.gitignore` 처리.

## 설치
```bash
pip install -r requirements.txt
# (가정용 PC에서 영상도 쓸 때만) pip install -r requirements-video.txt  + ffmpeg 설치(PATH)
cp .env.example .env      # GEMINI_API_KEY 를 채우기 (.env 는 커밋 금지)
```

## 사용
```bash
python -m bookpdf 입력.pdf                       # → output/책.pdf, output/전체원문.md, output/pages_md/, output/리포트.md
python -m bookpdf 입력.pdf --skip-ocr-if-text    # 텍스트 레이어가 있는 쪽은 OCR 생략(키 없이도 동작)
python -m bookpdf 책영상.mov --fps 3             # [로컬 전용] 영상 → 페이지 추출 → OCR → 책 PDF
python -m bookpdf 입력.pdf --max-pages 3         # 앞 3쪽만 시험
```
주요 옵션: `--out`, `--work`, `--name`(기본 `책.pdf`), `--title`, `--dpi`, `--threshold`(신뢰도 기준 0.7), `--page-thr`(영상 페이지 구분 민감도).

**중단 후 이어서 실행**: 같은 `--out`(또는 `--work`)로 다시 실행하면 `work/pages/pNNNN.json` 이 있는 쪽은 건너뜁니다.
실패한 쪽·키가 없어 건너뛴 쪽은 저장하지 않으므로 다음 실행에서 다시 처리됩니다. 호출 제한(429)은 서버가 알려준 시간만큼 기다린 뒤 재시도하고, `GEMINI_RPM` 으로 호출 간격도 제한합니다.

## 처리 순서
1. **입력 검증**: 파일 존재/암호/페이지 수(PDF), ffprobe 스트림·회전(영상).
2. **페이지 이미지화**: PDF → PyMuPDF 렌더. 영상 → ffmpeg(HEVC 디코딩, 회전 메타데이터 반영) 프레임 추출 →
   흐린 프레임 제거(Laplacian 분산) → 정지 구간 묶기 → 페이지 넘김 감지 → 페이지당 가장 선명한 1장.
3. **OCR**: Gemini에 이미지를 보내 Markdown(표는 Markdown 표) + 그림/도표 `box_2d` 를 받아, 좌표로 원본에서 그림을 잘라 저장.
4. **후처리**: 반복 머리글·쪽번호 제거, 겹친 인접 페이지 중복 문단 제거, 줄바꿈·하이픈 정리, 쪽 경계에서 끊긴 문단 병합.
5. **책 PDF**: Markdown → HTML → xhtml2pdf(NanumGothic). 표지·목차·본문(표·그림 포함).
6. **결과 검증**: 생성된 PDF를 다시 열어 글자·그림·NanumGothic 포함 여부 확인, `리포트.md` 에 신뢰도 낮은 쪽 목록.

## 검증 상태 (2026-10-26, 이 저장소의 테스트 기준)
| 항목 | 상태 |
|---|---|
| PDF 입력 → 책 PDF (텍스트 레이어 경로, 표·그림·목차·한글 폰트) | 합성 한국어 샘플로 **검증** |
| Gemini 경로(JSON 파싱, 429 재시도, 형식오류 재시도, 실패쪽 재개, 그림 크롭) | **가짜 클라이언트로만** 검증. 실제 Gemini 호출은 **미검증**(이 세션에 키 없음) |
| 기본 모델 `gemini-2.5-flash` | 공식 문서로 **확정하지 못함**. `GEMINI_MODEL` 로 교체. 모델명·무료 한도는 ai.google.dev 에서 확인 |
| `.mov` 영상 (H.264/HEVC, 회전 메타 90°, 흐림·손떨림·넘김 전환) | **합성 영상**으로만 검증. 실제 아이폰 영상은 **미검증** |
| Windows 에서의 실행 | **미검증**(Linux 에서만 실행). 폰트는 번들되어 있어 GTK 의존 없음 |

## 알려진 한계
- 영상: 한 페이지가 0.7초(2프레임@3fps) 미만으로 지나가면 누락될 수 있음 → `--fps` 를 올리세요. 같은 페이지가 둘로 쪼개지면 `--page-thr` 를 올리고, 다른 페이지가 합쳐지면 내리세요.
- 목차에 쪽번호는 없음(제목 목록 + PDF 책갈피). 
- 그림 크기는 원본에서 차지하던 폭 비율로 정해짐. 캡션은 Gemini가 준 경우에만 표시.
- xhtml2pdf 0.2.28 은 로컬 파일 읽기를 폴더로 제한하므로, 폰트·작업 폴더만 명시적으로 허용함(제한 해제 안 함).

## 테스트
```bash
pip install pytest
python -m pytest tests -q      # 영상 테스트는 ffmpeg + opencv 가 있을 때만 실행
```
