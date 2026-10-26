"""합성 테스트 영상 생성: 페이지를 N초씩 보여주고 사이에 넘김 전환/흐림/손떨림을 넣는다.
아이폰 세로 영상처럼 '가로로 저장 + 회전 메타데이터(display matrix)' 형태로 만든다."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pymupdf


def page_images(pdf: Path, h: int = 1280) -> list[np.ndarray]:
    out = []
    for p in pymupdf.open(pdf):
        pix = p.get_pixmap(dpi=150)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        w = int(arr.shape[1] * h / arr.shape[0]) // 2 * 2  # yuv420p 는 짝수 해상도 필요
        out.append(cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA))
    return out


def build(pdf: Path, out_mov: Path, codec: str = "libx264", rotation: int = 90, hold_s: float = 2.0,
          fps: int = 30) -> list[np.ndarray]:
    pages = page_images(pdf)
    frames_dir = out_mov.parent / "vframes"
    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True)
    n = 0
    rng = np.random.default_rng(0)

    def write(img: np.ndarray) -> None:
        nonlocal n
        # 저장은 가로: 시계방향 90° 회전해 저장 → 재생 시 반시계 90° 회전 메타데이터로 복원
        stored = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if rotation else img
        cv2.imwrite(str(frames_dir / f"{n:05d}.png"), stored)
        n += 1

    for k, pg in enumerate(pages):
        hold = int(hold_s * fps)
        for i in range(hold):
            dx, dy = (int(v) for v in rng.integers(-2, 3, 2))  # 손떨림(미세)
            fr = np.roll(pg, (dy, dx), axis=(0, 1))
            if i == 5:  # 초점 흔들림으로 흐린 프레임
                fr = cv2.GaussianBlur(fr, (0, 0), 6)
            write(fr)
        if k + 1 < len(pages):  # 페이지 넘김: 옆으로 밀리며 섞이는 전환 + 모션블러
            nxt = pages[k + 1]
            for t in range(1, 9):
                a = t / 9
                shift = int(a * pg.shape[1])
                canvas = np.roll(pg, -shift, axis=1).astype(np.float32) * (1 - a) + np.roll(nxt, pg.shape[1] - shift, axis=1) * a
                write(cv2.GaussianBlur(canvas.astype(np.uint8), (0, 0), 3))
    tmp = out_mov.with_suffix(".tmp.mov")
    enc = ["-c:v", codec] + (["-tag:v", "hvc1"] if codec == "libx265" else []) + ["-pix_fmt", "yuv420p", "-crf", "18"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"), *enc, str(tmp)], check=True)
    if rotation:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-display_rotation", str(rotation), "-i", str(tmp), "-c", "copy", str(out_mov)], check=True)
        tmp.unlink()
    else:
        tmp.rename(out_mov)
    shutil.rmtree(frames_dir, ignore_errors=True)
    return pages
