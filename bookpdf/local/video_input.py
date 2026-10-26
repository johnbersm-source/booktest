"""[로컬 전용] .mov(아이폰 세로 영상 포함) → 페이지당 최선의 프레임 1장.

가정용 PC에서만 실행한다. 필요: ffmpeg/ffprobe(PATH), opencv-python-headless, numpy.
단계(스트리밍): ffmpeg 가 디코딩한 프레임(회전 메타데이터 반영)을 파이프로 한 장씩 받아
      → 흐린 프레임 제외(Laplacian 분산) → 정지 구간 묶기(직전 프레임과 같은 페이지)
      → 페이지 넘김 감지 → 페이지당 가장 선명한 프레임 1장만 디스크에 저장.
      전체 프레임을 디스크/메모리에 쌓지 않으므로 300~400쪽 분량의 긴 영상도 처리할 수 있다.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

CMP_W = 960             # 비교용 축소 폭. 480 은 글자 한 자보다 팽창이 커져 줄 배치만 비교되어 실패(실측)
SAME_THR = 0.10         # 두 프레임의 '잉크 불일치 비율'이 이 값 미만이면 같은 페이지
                        # (합성 영상 실측: 같은 페이지 ≤0.004, 다른 페이지 ≥0.198. 실제 영상은 미검증 → --page-thr 로 조정)
BLUR_REL = 0.5          # 영상 전체 선명도 중앙값 대비 이 비율 미만이면 '흐림'
SINGLE_REL = 0.7        # 선명한 프레임이 1장뿐인 구간(빠른 넘김)은 중앙값의 이 비율 이상일 때만 페이지로 인정


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


def _target_dims(info: dict, max_side: int) -> tuple[int, int]:
    """회전 반영 후 출력 크기(짝수). 긴 변이 max_side 를 넘으면 축소."""
    w, h = int(info["width"]), int(info["height"])
    if abs(int(info.get("rotation", 0))) in (90, 270):
        w, h = h, w
    s = min(1.0, max_side / max(w, h))
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


def _read_exact(stream, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def _signature(video: Path, fps: float, same_thr: float, max_side: int) -> str:
    st = video.stat()
    return f"{st.st_size}:{st.st_mtime_ns}:{fps}:{same_thr}:{max_side}:v2"


def extract_pages(video: Path, out_dir: Path, fps: float = 3.0, same_thr: float = SAME_THR,
                  max_side: int = 2000, min_run: int = 1, log=print) -> tuple[list[Path], dict]:
    """영상 → 페이지 이미지(p0001.jpg …). 같은 영상·설정으로 이미 끝났으면 다시 하지 않는다."""
    if not video.is_file():
        raise ValueError(f"파일이 없습니다: {video}")
    if not 0.5 <= fps <= 10:
        raise ValueError("fps 는 0.5~10 사이여야 합니다(권장 2~5).")
    info = probe(video)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "_video_state.json"
    sig = _signature(video, fps, same_thr, max_side)
    if state_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        if st.get("signature") == sig and st.get("complete"):
            pages = sorted(out_dir.glob("p*.jpg"))
            if len(pages) == st["info"]["감지된 페이지"]:
                log(f"영상 처리 결과가 이미 있어 건너뜁니다({len(pages)}쪽).")
                return pages, st["info"]
    for old in out_dir.glob("p*.jpg"):
        old.unlink()
    state_path.unlink(missing_ok=True)

    W, H = _target_dims(info, max_side)
    cmd = [_need("ffmpeg"), "-v", "error", "-i", str(video),
           "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=W * H * 3 * 2)
    frame_bytes = W * H * 3
    expected = max(1, int(info["duration"] * fps))
    t0 = time.time()
    last_log = t0

    sharp_hist: deque[float] = deque(maxlen=300)
    n_frames = 0
    # 현재 정지 구간
    run_len = 0
    run_best_ok = None     # (선명도, 프레임, 축소본) — 흐리지 않은 것 중 최선
    run_best_any = None    # 흐린 것 포함 최선(폴백)
    prev_small = None
    # 지금까지 확정된 페이지
    pages: list[Path] = []
    last_small = None
    last_sharp = 0.0
    n_runs = 0

    def write(n: int, frame: np.ndarray) -> Path:
        p = out_dir / f"p{n:04d}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return p

    def close_run() -> None:
        nonlocal last_small, last_sharp, n_runs
        if run_len < 2:
            # 한 프레임짜리 구간: 빠르게 넘기는 영상에서 선명한 샘플이 1장뿐인 쪽일 수 있다.
            # 전환(페이지 넘김) 중 흐린 프레임이 가짜 쪽이 되지 않도록 선명한 것만 인정한다.
            med = float(np.median(sharp_hist)) if sharp_hist else 0.0
            if min_run > 1 or run_best_ok is None or run_best_ok[0] < SINGLE_REL * med:
                return
        elif run_len < min_run:
            return
        n_runs += 1
        sharp, frame, sm = run_best_ok or run_best_any
        if last_small is not None and page_diff(last_small, sm) < same_thr:
            if sharp > last_sharp:          # 같은 페이지로 판정 → 더 선명한 쪽으로 교체
                write(len(pages), frame)
                last_small, last_sharp = sm, sharp
            return
        pages.append(write(len(pages) + 1, frame))
        last_small, last_sharp = sm, sharp

    try:
        while True:
            raw = _read_exact(proc.stdout, frame_bytes)
            if raw is None:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(H, W, 3)
            n_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = _sharpness(gray)
            sm = _small(gray)
            sharp_hist.append(sharp)
            blur_thr = BLUR_REL * float(np.median(sharp_hist)) if len(sharp_hist) >= 10 else 0.0
            if run_len and prev_small is not None and page_diff(prev_small, sm) < same_thr:
                run_len += 1
            else:
                close_run()
                run_len, run_best_ok, run_best_any = 1, None, None
            prev_small = sm
            cand = (sharp, frame.copy(), sm)
            if run_best_any is None or sharp > run_best_any[0]:
                run_best_any = cand
            if sharp >= blur_thr and (run_best_ok is None or sharp > run_best_ok[0]):
                run_best_ok = cand
            now = time.time()
            if now - last_log >= 20:
                last_log = now
                done = min(1.0, n_frames / expected)
                rate = n_frames / max(1e-6, now - t0)
                eta = (expected - n_frames) / max(1e-6, rate)
                log(f"  영상 분석 {done:5.1%} ({n_frames}/{expected}프레임, {rate:.1f}프레임/초) "
                    f"페이지 {len(pages)}개 감지, 남은 약 {eta / 60:.1f}분")
        close_run()
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        proc.stderr.close()
        rc = proc.wait()
    if rc != 0 and n_frames == 0:
        raise RuntimeError(f"ffmpeg 프레임 추출 실패(코드 {rc}): {err.strip()[:400]}")
    if not pages:
        if n_frames == 0:
            raise RuntimeError("추출된 프레임이 없습니다.")
        raise RuntimeError("정지 구간(페이지)을 찾지 못했습니다. --fps 를 올리거나 --page-thr 를 확인하세요.")
    info.update({
        "출력 프레임 크기(WxH)": f"{W}x{H}", "분석 프레임": n_frames, "정지 구간": n_runs,
        "감지된 페이지": len(pages), "페이지 구분 기준(same_thr)": same_thr,
        "선명도 중앙값": round(float(np.median(sharp_hist)), 1) if sharp_hist else 0,
        "분석 소요(초)": round(time.time() - t0, 1),
    })
    state_path.write_text(json.dumps({"signature": sig, "complete": True, "info": info}, ensure_ascii=False), encoding="utf-8")
    return pages, info
