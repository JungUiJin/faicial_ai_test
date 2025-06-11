import os
import math
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity as ssim
import numpy as np

# === 얼굴 부위별 랜드마크 인덱스 ===
FACE_PARTS = {
    "left_eye": [33, 133, 160, 159, 158, 157, 173],
    "right_eye": [362, 263, 387, 386, 385, 384, 398],
    "nose": [1, 2, 98, 327],
    "mouth": [13, 14, 78, 308, 61, 291],
    "left_ear": [234, 93],
    "right_ear": [454, 323],
    "left_chin": [152, 150, 149, 176],
    "right_chin": [152, 379, 378, 400],
}

# === 패딩 비율 설정 ===
PADDING_RATIO_MAP = {
    'left_eye': {'top': 0.02, 'bottom': 0.02, 'left': 0.04, 'right': 0.04},
    'right_eye': {'top': 0.02, 'bottom': 0.02, 'left': 0.04, 'right': 0.04},
    'nose': {'top': 0.1, 'bottom': 0.03, 'left': 0.03, 'right': 0.03},
    'mouth': {'top': 0.05, 'bottom': 0.06, 'left': 0.04, 'right': 0.04},
    'left_ear': {'top': 0.1, 'bottom': 0.08, 'left': 0.10, 'right': 0.00},
    'right_ear': {'top': 0.1, 'bottom': 0.08, 'left': 0.00, 'right': 0.10},
    "left_chin": {'top': 0.12, 'bottom': 0.02, 'left': 0.10, 'right': 0.00},
    "right_chin": {'top': 0.12, 'bottom': 0.02, 'left': 0.00, 'right': 0.10},
}

# ✅ 추가: 눈 좌표 기준 이미지 정렬 함수
def align_face_by_eyes(image: Image.Image, left_eye: tuple, right_eye: tuple) -> Image.Image:
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    return image.rotate(-angle, resample=Image.BICUBIC, center=(image.width // 2, image.height // 2))

# ✅ 추가: 평균 좌표 계산 함수
def average_point(points):
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)

# 영역별 자르기
def devide_region(image_pil: Image.Image, landmarks: list[tuple], indices: list[int], padding_ratio: dict) -> Image.Image:
    points = [landmarks[i] for i in indices if 0 <= i < len(landmarks)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    width, height = image_pil.size

    top = int(padding_ratio.get('top', 0.02) * height)
    bottom = int(padding_ratio.get('bottom', 0.02) * height)
    left = int(padding_ratio.get('left', 0.02) * width)
    right = int(padding_ratio.get('right', 0.02) * width)

    min_x = max(min(xs) - left, 0)
    max_x = min(max(xs) + right, width)
    min_y = max(min(ys) - top, 0)
    max_y = min(max(ys) + bottom, height)

    return image_pil.crop((min_x, min_y, max_x, max_y))

# ✅ 수정: 정렬된 이미지에서 부위 추출
def get_face_parts(landmarks: list[tuple], image_pil: Image.Image) -> dict[str, Image.Image]:
    # === 정렬 전 눈 중심 계산
    left_eye_center = average_point([landmarks[i] for i in FACE_PARTS["left_eye"]])
    right_eye_center = average_point([landmarks[i] for i in FACE_PARTS["right_eye"]])

    # === 이미지 정렬
    aligned_image = align_face_by_eyes(image_pil, left_eye_center, right_eye_center)

    parts = {}
    for part_name, indices in FACE_PARTS.items():
        padding_ratio = PADDING_RATIO_MAP.get(part_name, {})
        cropped = devide_region(aligned_image, landmarks, indices, padding_ratio)
        parts[part_name] = cropped
    return parts

# SSIM 비교 함수 (좌우 반전 포함)
def compare_ssim_flipped_images(img1: Image.Image, img2: Image.Image) -> float:
    img1 = img1.convert("L")
    img2 = img2.convert("L")
    img1 = ImageOps.mirror(img1)

    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    arr1 = np.array(img1)
    arr2 = np.array(img2)
    score, _ = ssim(arr1, arr2, full=True)
    return round(score * 100, 2)

def compare_split_match(image: Image.Image) -> tuple[float, Image.Image, Image.Image]:
    width, height = image.size
    mid = width // 2
    left_half = image.crop((0, 0, mid, height))
    right_half = image.crop((mid, 0, width, height))
    right_half_flipped = ImageOps.mirror(right_half)

    if left_half.size != right_half_flipped.size:
        right_half_flipped = right_half_flipped.resize(left_half.size)

    arr1 = np.array(left_half.convert("L"))
    arr2 = np.array(right_half_flipped.convert("L"))
    score, _ = ssim(arr1, arr2, full=True)

    return round(score * 100, 2), left_half, right_half

def weighted_average(score_dict, weights):
    total_weighted_score = 0.0
    total_weight = 0.0
    for part, score in score_dict.items():
        if score is not None:
            weight = weights.get(part, 1.0)
            total_weighted_score += score * weight
            total_weight += weight
    return round(total_weighted_score / total_weight, 2) if total_weight else None

# 일치율 계산
def compare_match_parts_from_images(parts: dict[str, Image.Image]) -> dict[str, float | None]:
    results: dict[str, float | None] = {}

    results["eyes"] = (
        compare_ssim_flipped_images(parts["left_eye"], parts["right_eye"])
        if "left_eye" in parts and "right_eye" in parts else None
    )

    results["ears"] = (
        compare_ssim_flipped_images(parts["left_ear"], parts["right_ear"])
        if "left_ear" in parts and "right_ear" in parts else None
    )

    if "nose" in parts:
        score, left_nose, right_nose = compare_split_match(parts["nose"])
        results["nose"] = score
        parts["left_nose"] = left_nose
        parts["right_nose"] = right_nose
        del parts["nose"]

    if "mouth" in parts:
        score, left_mouth, right_mouth = compare_split_match(parts["mouth"])
        results["mouth"] = score
        parts["left_mouth"] = left_mouth
        parts["right_mouth"] = right_mouth
        del parts["mouth"]

    results["chin"] = (
        compare_ssim_flipped_images(parts["left_chin"], parts["right_chin"])
        if "left_chin" in parts and "right_chin" in parts else None
    )

    return results
