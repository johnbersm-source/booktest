import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import make_spread as ms  # noqa: E402
from bookpdf import spread  # noqa: E402


# (안쪽여백px, 휨px, 기울기°, 제본그림자) — 마지막 칸 True 는 '다른 쪽 표식이 약간 섞여도 되는' 어려운 조합
CASES = [
    (70, 0, 0.0, True), (70, 0, 1.5, True), (70, 0, -2.0, True),
    (30, 24, 1.5, True), (35, 20, -1.5, True),
    (40, 0, 2.0, False), (30, 0, 0.0, False), (60, 0, -2.0, False),
]


@pytest.mark.parametrize("inner,bow,tilt,shadow", CASES)
@pytest.mark.parametrize("seed", [1, 3])
def test_split_order_and_no_clipping(inner, bow, tilt, shadow, seed):
    img, _ = ms.make_spread(seed=seed, tilt_deg=tilt, inner_margin=inner, bow_px=bow,
                            marker_y=(30, 925), gutter_shadow=shadow)
    ok, g = spread.is_spread(img, "auto")
    assert ok and g.found, g
    assert 0.46 < g.x < 0.56
    left, right = spread.split_image(img, g)
    red_total, blue_total = ms.count_color(img, "red"), ms.count_color(img, "blue")
    assert red_total > 500 and blue_total > 500
    assert ms.count_color(left, "red") >= 0.99 * red_total, "왼쪽 쪽 표식이 잘림"
    assert ms.count_color(right, "blue") >= 0.99 * blue_total, "오른쪽 쪽 표식이 잘림"
    # 좌우가 뒤바뀌거나 반대쪽이 통째로 섞이면 실패. 겹침 구간의 가장자리 표식 일부(≤25%)는 허용
    assert ms.count_color(left, "blue") <= 0.35 * blue_total, "왼쪽 크롭에 오른쪽 쪽 내용이 많이 섞임"
    assert ms.count_color(right, "red") <= 0.25 * red_total, "오른쪽 크롭에 왼쪽 쪽 내용이 많이 섞임"


def test_extreme_geometry_never_clips_but_may_mix():
    """휨 40px + 기울기 3° + 안쪽 여백 1.8%: 세로 분할의 한계로 반대쪽 가장자리가 섞일 수 있으나 글자는 잘리지 않아야 한다."""
    img, _ = ms.make_spread(seed=1, tilt_deg=3.0, inner_margin=25, bow_px=40, marker_y=(30, 925))
    ok, g = spread.is_spread(img, "auto")
    assert ok
    left, right = spread.split_image(img, g)
    assert ms.count_color(left, "red") >= 0.99 * ms.count_color(img, "red")
    assert ms.count_color(right, "blue") >= 0.99 * ms.count_color(img, "blue")


def test_wrong_gutter_is_detected_by_the_check():
    """검사 자체의 민감도: 제본선을 일부러 크게 틀리면 위 단언이 실패해야 한다."""
    img, _ = ms.make_spread(seed=3)
    _, g = spread.is_spread(img, "auto")
    left, right = spread.split_image(img, g, offset=0.06)
    assert ms.count_color(left, "blue") > 0.35 * ms.count_color(img, "blue")


def test_single_page_passthrough_and_modes():
    page, _ = ms.page_img(5)
    single = page
    assert spread.is_spread(single, "auto")[0] is False       # 세로 이미지는 그대로
    spr, _ = ms.make_spread(seed=4)
    assert spread.is_spread(spr, "off")[0] is False
    assert spread.is_spread(spr, "on")[0] is True


def test_landscape_single_page_not_split():
    # 가로 슬라이드처럼 가운데에 빈 띠가 없는 가로 이미지는 auto 에서 분리하지 않는다
    from PIL import ImageDraw
    im = Image.new("RGB", (1400, 1000), "white")
    d = ImageDraw.Draw(im)
    for y in range(60, 940, 30):
        d.line([(60, y), (1340, y)], fill="black", width=5)
    assert spread.is_spread(im, "auto")[0] is False


def test_offset_correction_shifts_cut():
    img, _ = ms.make_spread(seed=6)
    _, g = spread.is_spread(img, "auto")
    l0, r0 = spread.split_image(img, g, offset=0.0)
    l1, r1 = spread.split_image(img, g, offset=0.05)
    assert l1.width > l0.width and r1.width < r0.width


def test_expand_spreads_files_and_resume(tmp_path):
    a, _ = ms.make_spread(seed=1)
    b, _ = ms.page_img(9)
    pa, pb = tmp_path / "p0001.png", tmp_path / "p0002.png"
    a.save(pa); b.save(pb)
    out, info = spread.expand_spreads([pa, pb], tmp_path / "split", "auto")
    assert [i["kind"] for i in info] == ["left", "right", "single"]
    assert [p.name for p in out] == ["p0001.png", "p0002.png", "p0003.png"]
    m = out[0].stat().st_mtime_ns
    out2, _ = spread.expand_spreads([pa, pb], tmp_path / "split", "auto")
    assert out2[0].stat().st_mtime_ns == m   # 이미 있는 파일은 다시 쓰지 않음
