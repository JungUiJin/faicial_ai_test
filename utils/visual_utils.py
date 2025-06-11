from PIL import Image, ImageDraw
from typing import List, Tuple

def draw_landmark_points(
    image: Image.Image,
    landmarks: List[Tuple[float, float]],
    color: str = "lime",
    radius: int = 3
) -> Image.Image:
    """
    디버깅용: PIL Image 위에 랜드마크 좌표마다 작은 원(circle)을 그려 반환합니다.

    Args:
        image: PIL Image 객체 (RGBA 모드 권장)
        landmarks: [(x, y), ...] 형태의 랜드마크 좌표 리스트
        color: 원의 색상 (기본 'lime')
        radius: 원의 반지름(px) (기본값 3)

    Returns:
        랜드마크가 오버레이된 새로운 PIL Image 객체
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for x, y in landmarks:
        left_up = (x - radius, y - radius)
        right_down = (x + radius, y + radius)
        draw.ellipse([left_up, right_down], fill=color)

    return Image.alpha_composite(image, overlay)

def draw_specific_points(
    image: Image.Image,
    landmarks: List[Tuple[float, float]],
    indices: List[int],
    color: str = "red",
    radius: int = 6
) -> Image.Image:
    """
    지정된 인덱스의 랜드마크 좌표만 강조 표시합니다.

    Args:
        image: PIL Image 객체 (RGBA 모드 권장)
        landmarks: [(x, y), ...] 형태의 랜드마크 좌표 리스트
        indices: 강조할 랜드마크 인덱스 리스트
        color: 원의 색상 (기본 'red')
        radius: 원의 반지름(px) (기본값 6)

    Returns:
        강조된 랜드마크 오버레이된 PIL Image 객체
    """
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for idx in indices:
        if idx < len(landmarks):
            x, y = landmarks[idx]
            left_up = (x - radius, y - radius)
            right_down = (x + radius, y + radius)
            draw.ellipse([left_up, right_down], fill=color)

    return Image.alpha_composite(image, overlay)
