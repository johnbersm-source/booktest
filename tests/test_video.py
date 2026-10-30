"""로컬 전용 영상 모듈 테스트(합성 영상). ffmpeg/opencv 가 없으면 건너뛴다. 실제 아이폰 영상 검증은 아님."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    pytest.skip("ffmpeg 없음", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).parent))
import make_sample  # noqa: E402
import make_video  # noqa: E402
from bookpdf.local import video_input  # noqa: E402


def _g(img):
    return video_input._small(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


@pytest.mark.parametrize("codec", ["libx264", "libx265"])
def test_video_pages_orientation_and_order(tmp_path, codec):
    pdf = tmp_path / "s.pdf"
    make_sample.build(pdf)
    mov = tmp_path / "v.mov"
    enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    if codec not in enc:
        pytest.skip(f"ffmpeg 에 {codec} 인코더 없음")
    pages = make_video.build(pdf, mov, codec=codec, rotation=90)  # 실패하면 테스트 실패로 드러나야 함
    info = video_input.probe(mov)
    assert info["rotation"] in (90, -90, 270) and info["width"] > info["height"]  # 가로 저장 + 회전 메타

    out, stats = video_input.extract_pages(mov, tmp_path / "work" / "page_images", fps=3)
    assert len(out) == len(pages), stats
    assert stats["감지된 페이지"] == len(pages)
    for k, p in enumerate(out):
        img = cv2.imread(str(p))
        assert img.shape[0] > img.shape[1], "회전 메타데이터가 반영되지 않아 가로로 나옴"
        diffs = [video_input.page_diff(_g(img), _g(pg)) for pg in pages]
        assert diffs[k] < 0.05, f"p{k+1} 이 원본 페이지와 일치하지 않음: {diffs}"
        assert all(d >= video_input.SAME_THR for j, d in enumerate(diffs) if j != k), f"다른 페이지와 구분 안 됨: {diffs}"
