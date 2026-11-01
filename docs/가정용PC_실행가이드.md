# 📱 가정용 PC에서 .mov → 책 PDF (모바일용 요약)

> 이 도구는 **영상 → 책 PDF** 변환을 **내 PC에서** 돌립니다. 영상은 GitHub에 올리지 않습니다.
> 순서: ① 설치 → ② 키 설정 → ③ 3쪽 시험 → ④ 전체 실행 → ⑤ 이어하기

---

## ① 한 번만: 설치

**공통 준비물**
- Python 3.10 이상
- ffmpeg (영상 해독 + HEVC 아이폰 영상 대응)
- 이 저장소 내려받기 (`git clone` 또는 ZIP)

**맥북 M1 (macOS)**
```bash
brew install ffmpeg python
cd booktest
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-video.txt
```

**N100 미니PC (Windows 기준)**
```bat
winget install Gyan.FFmpeg
cd booktest
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt -r requirements-video.txt
```
- 설치 후 새 터미널에서 `ffmpeg -version` 이 나오면 OK.
- 회사/학교 네트워크에서는 막힐 수 있습니다. **막히면 우회하지 말고** 막힌 요청을 그대로 알려 주세요.

> Linux면 `sudo apt install ffmpeg` 후 맥과 같은 방식입니다.
> (이 가이드의 Windows/macOS 명령은 이 세션에서 직접 실행해 보지 못했습니다. 실행 중 오류가 나면 오류 문장을 그대로 알려 주세요.)

---

## ② 키 설정 (Gemini)

1. https://aistudio.google.com/apikey 에서 API 키 발급
2. 저장소 폴더에서 `.env.example` 을 `.env` 로 복사
3. `.env` 를 열어 채우기

```
GEMINI_API_KEY=발급받은키
GEMINI_MODEL=gemini-2.5-flash
GEMINI_RPM=10
GEMINI_RPD=0
```
- `GEMINI_MODEL`: 모델명은 자주 바뀝니다. 공식 문서(ai.google.dev)에서 현재 무료로 쓸 수 있는 모델로 바꾸세요.
- `GEMINI_RPM`: 분당 호출 상한. 무료 한도에 맞춰 낮추면 429 대기가 줄어듭니다.
- `GEMINI_RPD`: **하루 호출 상한**. 한도를 아는 경우 그 값 이하로 두면, 한도 직전에 저장하고 멈춥니다(0 = 제한 없음).
- ⚠️ `.env` 는 절대 올리거나 공유하지 마세요(이미 `.gitignore` 처리).

---

## ③ 먼저 3쪽만 시험 (필수)

```bash
python -m bookpdf 책영상.mov --max-pages 3 --out output_test
```
확인할 것
- `output_test/책.pdf` 열기 → 글자 / 쪽 순서
- `output_test/리포트.md` → 펼침면이 제대로 갈렸는지, 쪽번호 점검

---

## ④ 전체 실행

```bash
python -m bookpdf 책영상.mov --out output
```
화면에 이렇게 나옵니다
- `영상 분석 42% … 남은 약 3.1분`
- `원본 210장 → 책 400쪽 (펼침면 190장 분리)`
- `[57/400] … 쪽당 평균 6.1초, 남은 343쪽 약 35분(예상)`

**오래 걸려도 정상입니다.** 400쪽은 호출 제한 때문에 수 시간~며칠 걸릴 수 있습니다.

---

## ⑤ 중단됐을 때 — 그냥 같은 명령 다시

```bash
python -m bookpdf 책영상.mov --out output
```
- 끝난 쪽은 건너뛰고 **남은 쪽만** 처리합니다.
- 하루 한도에 닿으면 `중단: … 같은 명령을 다시 실행` 이 나오고 책 PDF는 만들지 않습니다.
  다음 날 같은 명령을 다시 실행하세요. (지금까지 분량만 보려면 `--allow-partial`)
- 설정(`--spread`, `--fps` 등)을 바꾸면 **이전 결과와 섞이지 않게** 멈춥니다.
  처음부터 다시 하려면 `--restart`.

---

## 🛠 자주 쓰는 옵션

| 상황 | 옵션 |
|---|---|
| 펼침면이 안 갈라짐 / 단일 쪽을 갈라버림 | 먼저 `--split-only` 로 분리만 시험(아래) → `--spread-thr` 를 낮춤(0.4, 0.3) / 올림(0.6). 그래도 안 되면 `--spread on` / `--spread off` |
| 갈라진 선이 살짝 어긋남 | `--gutter-offset 0.02` (오른쪽 2%) / `-0.02` |
| 쪽이 빠짐(빨리 넘긴 영상) | `--fps 5` |
| 같은 쪽이 둘로 쪼개짐 | `--page-thr 0.15` (올리기) |
| 다른 쪽이 합쳐짐 | `--page-thr 0.07` (내리기) |
| 메모리가 모자람 | `--max-frame-side 1600` |
| 하루 호출 수 제한 | `--max-calls 200` |

## 📋 끝난 뒤 볼 것
1. `output/책.pdf`
2. `output/리포트.md` → **쪽번호 점검**(누락/중복 의심 쪽), **신뢰도 낮은 쪽**
3. 의심 쪽 원본 이미지: `output/work/page_images/pNNNN.jpg`

## ⚠️ 한계 (솔직하게)
- 실제 아이폰 영상으로는 아직 검증하지 못했습니다(합성 영상으로만 확인). 3쪽 시험에서 결과를 꼭 확인하세요.
- 제본선이 심하게 휘거나 기울면 반대쪽 가장자리가 조금 섞일 수 있습니다(글자 잘림보다 섞임을 택한 설계).
- 쪽번호 점검은 OCR이 번호를 읽어야 동작합니다. 번호가 없는 책이면 점검이 생략됩니다.

## 🆕 Gemini 한도(429)에 막힐 때 → tesseract 로 전체 책
```
python -m bookpdf 책.pdf --ocr tesseract --dpi 300 --pages 114-215 --embed-page-images
```
- 한도 없음. 글자 인식은 Gemini보다 거칠 수 있어 `리포트.md` 의 신뢰도 낮은 쪽을 확인하세요(실제 책 품질은 미검증).
- 차트 위주 쪽은 원본 이미지로 들어갑니다(`--embed-min-chars`/`--embed-min-ink` 로 기준 조정).
- tesseract 설치는 README 의 'tesseract 모드' 참고(Windows/macOS 는 미검증).
- Gemini 가 `일일 한도(RPD)` 로 멈추면 `--max-calls` 안내가 출력됩니다. 내일(미국 태평양시 자정 이후) 같은 명령으로 이어집니다.

## 펼침면이 제대로 갈렸는지 먼저 확인하기 (`--split-only`)

OCR 전에 분리만 시험할 수 있습니다. API 키·tesseract 가 없어도 됩니다.

```
python -m bookpdf IMG_5899_chart.pdf --out output --split-only
```

1. `output/분리점검.md` 를 열어 요약(원본 N장 중 K장 분리)과 점검 대상 프레임 목록을 봅니다.
2. `output/work/split_only/preview/preview_01.jpg` … 에서 프레임을 눈으로 확인합니다(빨강=분리선, 주황=주변 프레임 기준으로 분리, 회색 점선=단일로 판정했지만 제본선 후보가 있던 위치, `*`=경계).
3. 프레임별 점수·사유는 `output/work/split_only/page_images/spread_log.csv` (엑셀). 두 쪽이 한 장으로 남은 프레임의 점수를 보고 `--spread-thr` 를 그 아래로 내립니다.
4. `--spread-thr 0.4 --split-only` 처럼 값을 바꿔 다시 실행하면 `split_only` 폴더만 새로 만들어집니다(본 실행 결과는 그대로).
5. 만족하면 `--split-only` 를 뺀 같은 명령으로 실행합니다. 이전 판정으로 만든 `work` 가 있으면 `--restart` 가 필요합니다(판정 방식이 바뀐 버전이라 설정 불일치로 멈춥니다).

쪽번호가 256→258→260→263 처럼 건너뛰면 `리포트.md` 의 쪽번호 점검이 원인을 나눠 보여줍니다.
- **분리 실패 의심**: 그 구간의 이미지가 단일로 처리됐는데 제본 띠/그림자 증거가 있거나 가로 이미지 → 한 이미지에 두 쪽이 들어 있을 가능성. `--spread-thr` 를 낮추세요.
- **촬영 누락 의심**: 분리 실패로 설명되지 않는 건너뜀 → 영상에서 쪽을 놓쳤을 가능성. `--fps` 를 올리거나 쪽을 천천히 넘겨 다시 찍으세요.
(추정입니다. 번호 인식 오류가 끼면 틀릴 수 있으니 `work/page_images/` 이미지를 직접 확인하세요.)
