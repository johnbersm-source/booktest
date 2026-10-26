"""[로컬 전용] .mov(아이폰 세로 영상 포함) → 페이지당 최선의 프레임 1장.

가정용 PC에서만 실행한다. 필요: ffmpeg/ffprobe(PATH), opencv-python-headless, numpy.
단계: ffmpeg 프레임 추출(회전 메타데이터 반영) → 흐린 프레임 제거(Laplacian 분산)
      → 정지 구간 묶기(직전 프레임과 거의 동일) → 화면 변화량으로 페이지 넘김 감지
      → 페이지당 가장 선명한 프레임 1장.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

CMP_W = 960             # 비교용 축소 폭. 480 은 글자 한 자보다 팽창이 커져 줄 배치만 비교되어 실패(실측)
SAME_THR = 0.10         # 두 프레임의 '잉크 불일치 비율'이 이 값 미만이면 같은 페이지
                        # (합성 영상 실측: 같은 페이지 ≤0.004, 다른 페이지 ≥0.198. 실제 영상은 미검증 → --page-thr 로 조정)
BLUR_REL = 0.5          # 영상 전체 선명도 중앙값 대비 이 비율 미만이면 '흐림'


def _need(tool: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise RuntimeError(f"{tool} 를 찾을 수 없습니다. ffmpeg 를 설치하고 PATH 에 추가하세요.")
    return p


def probe(video: Path) -> dict:
    out = subprocess.run(
        [_need("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
         "-of", "json", str(video)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe 실패: {out.stderr.strip()[:300]}")
    data = json.loads(out.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise ValueError("영상 스트림을 찾지 못했습니다.")
    st = streams[0]
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(st.get("tags", {}).get("rotate", rot) or rot)
    return {"codec": st.get("codec_name"), "width": st.get("width"), "height": st.get("height"),
            "rotation": rot, "duration": float(data.get("format", {}).get("duration", 0) or 0)}


def extract_frames(video: Path, frame_dir: Path, fps: float) -> list[Path]:
    """ffmpeg 로 fps 추출. 회전 메타데이터(display matrix)는 ffmpeg 의 autorotate 로 반영한다.
    HEVC 디코딩도 ffmpeg 가 처리한다."""
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old in frame_dir.glob("f*.jpg"):
        old.unlink()
    cmd = [_need("ffmpeg"), "-v", "error", "-y", "-i", str(video),
           "-vf", f"fps={fps}", "-q:v", "2", str(frame_dir / "f%06d.jpg")]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 프레임 추출 실패: {r.stderr.strip()[:400]}")
    frames = sorted(frame_dir.glob("f*.jpg"))
    if not frames:
        raise RuntimeError("추출된 프레임이 없습니다.")
    return frames


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _small(gray: np.ndarray) -> np.ndarray:
    h = max(1, int(gray.shape[0] * CMP_W / gray.shape[1]))
    return cv2.resize(gray, (CMP_W, h), interpolation=cv2.INTER_AREA)


def _ink(g: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12) > 0


def page_diff(ga: np.ndarray, gb: np.ndarray) -> float:
    """두 축소 회색조 프레임이 '다른 페이지'일 정도: 0(같음)~1(전혀 다름).
    카메라 평행이동은 위상상관으로 정렬해 제거하고, 글자(잉크) 픽셀이 상대 프레임의
    (3x3 팽창한) 잉크와 겹치지 않는 비율을 센다. 전체 평균 밝기 차이는 글자 페이지끼리
    거의 구분되지 않아 쓰지 않는다."""
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ga.astype(np.float32), gb.astype(np.float32), win)
    gb = cv2.warpAffine(gb, np.float32([[1, 0, -dx], [0, 1, -dy]]), (gb.shape[1], gb.shape[0]),
                        borderMode=cv2.BORDER_REPLICATE)
    a, b = _ink(ga), _ink(gb)
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 and nb == 0:
        return 0.0
    k = np.ones((3, 3), np.uint8)
    da = cv2.dilate(a.astype(np.uint8), k) > 0
    db = cv2.dilate(b.astype(np.uint8), k) > 0
    return float(max((a & ~db).sum() / max(1, na), (b & ~da).sum() / max(1, nb)))


def select_pages(frames: list[Path], min_run: int = 2, same_thr: float = SAME_THR) -> tuple[list[int], dict]:
    """프레임 인덱스 목록(페이지당 1개)과 통계를 돌려준다. 메모리를 아끼려 스트리밍으로 처리한다."""
    # 1패스: 선명도(흐림 기준용)
    sharp: list[float] = []
    for p in frames:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        sharp.append(_sharpness(g) if g is not None else 0.0)
    valid = [s for s in sharp if s > 0]
    med = float(np.median(valid)) if valid else 1.0
    blur_thr = med * BLUR_REL

    # 2패스: 정지 구간(직전 프레임과 같은 페이지로 보이는 연속 프레임) 묶기
    runs: list[tuple[list[int], np.ndarray]] = []   # (구간 프레임 인덱스, 대표 프레임 축소본)
    cur: list[int] = []
    cur_small: dict[int, np.ndarray] = {}
    prev_small = None

    def flush():
        if len(cur) >= min_run:
            ok = [i for i in cur if sharp[i] >= blur_thr] or cur
            b = max(ok, key=lambda i: sharp[i])
            runs.append((list(cur), cur_small[b]))
    for i, p in enumerate(frames):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        sm = _small(g)
        if cur and prev_small is not None and page_diff(prev_small, sm) < same_thr:
            cur.append(i)
        else:
            flush()
            cur, cur_small = [i], {}
        cur_small[i] = sm
        # 메모리: 구간 안에서는 가장 선명한 프레임의 축소본만 유지
        if len(cur_small) > 2:
            keep = max(cur_small, key=lambda j: sharp[j] if sharp[j] >= blur_thr else -1)
            cur_small = {j: s for j, s in cur_small.items() if j in (keep, i)}
        prev_small = sm
    flush()
    if not runs:  # 정지 구간이 없으면 가장 선명한 프레임 1장으로 폴백
        best = max(range(len(frames)), key=lambda i: sharp[i])
        return [best], {"추출 프레임": len(frames), "정지 구간": 0, "감지된 페이지": 1,
                        "경고": "정지 구간을 찾지 못해 1장만 선택(--fps 를 올리거나 --page-thr 확인)"}

    # 인접 구간 병합(손으로 잠깐 가린 경우 등) + 페이지당 가장 선명한 프레임
    pages: list[int] = []
    page_small: list[np.ndarray] = []
    for idxs, sm in runs:
        ok = [i for i in idxs if sharp[i] >= blur_thr] or idxs
        b = max(ok, key=lambda i: sharp[i])
        if page_small and page_diff(page_small[-1], sm) < same_thr:
            if sharp[b] > sharp[pages[-1]]:
                pages[-1], page_small[-1] = b, sm
            continue
        pages.append(b)
        page_small.append(sm)
    stats = {
        "추출 프레임": len(frames), "정지 구간": len(runs), "감지된 페이지": len(pages),
        "선명도 중앙값": round(med, 1), "흐림 기준": round(blur_thr, 1),
        "선택 프레임 선명도": [round(sharp[i], 1) for i in pages],
        "페이지 구분 기준(same_thr)": same_thr,
    }
    return pages, stats


def extract_pages(video: Path, out_dir: Path, fps: float = 3.0, same_thr: float = SAME_THR) -> tuple[list[Path], dict]:
    if not video.is_file():
        raise ValueError(f"파일이 없습니다: {video}")
    if not 0.5 <= fps <= 10:
        raise ValueError("fps 는 0.5~10 사이여야 합니다(권장 2~5).")
    info = probe(video)
    frames = extract_frames(video, out_dir.parent / "frames", fps)
    idx, stats = select_pages(frames, same_thr=same_thr)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for n, i in enumerate(idx, start=1):
        dst = out_dir / f"p{n:04d}.jpg"
        shutil.copyfile(frames[i], dst)
        pages.append(dst)
    first = cv2.imread(str(pages[0]))
    info.update(stats)
    info["출력 프레임 크기(WxH)"] = f"{first.shape[1]}x{first.shape[0]}"
    return pages, info
