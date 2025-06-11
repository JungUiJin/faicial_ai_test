import os
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

# 비율 기반 패딩 설정 (비율: 0.0 ~ 1.0)
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

# 영역별 자르기 함수
def devide_region(image_pil: Image.Image, landmarks: list[tuple], indices: list[int], padding_ratio: dict) -> Image.Image:
    points = [landmarks[i] for i in indices if 0 <= i < len(landmarks)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    width, height = image_pil.size

    # 비율 기반 padding 계산
    top = int(padding_ratio.get('top', 0.02) * height)
    bottom = int(padding_ratio.get('bottom', 0.02) * height)
    left = int(padding_ratio.get('left', 0.02) * width)
    right = int(padding_ratio.get('right', 0.02) * width)

    min_x = max(min(xs) - left, 0)
    max_x = min(max(xs) + right, width)
    min_y = max(min(ys) - top, 0)
    max_y = min(max(ys) + bottom, height)

    return image_pil.crop((min_x, min_y, max_x, max_y))

# 얼굴 부위별 검출
def get_face_parts(landmarks: list[tuple], image_pil: Image.Image) -> dict[str, Image.Image]:
    
    corrected_image = correct_face_tilt(image_pil)
    
    parts = {}
    for part_name, indices in FACE_PARTS.items():
        padding_ratio = PADDING_RATIO_MAP.get(part_name, {})
        cropped = devide_region(corrected_image, landmarks, indices, padding_ratio)
        parts[part_name] = cropped
    return parts

def correct_face_tilt(pil_image):
    # PIL 이미지를 numpy 배열로 변환
    image_np = np.array(pil_image)

    # mediapipe 얼굴 탐지 초기화
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_np)

        if not results.multi_face_landmarks:
            print("얼굴을 찾을 수 없습니다.")
            return pil_image

        face_landmarks = results.multi_face_landmarks[0]

        # 양쪽 눈의 중심 좌표 얻기 (mediapipe의 눈 좌표 참조)
        left_eye = face_landmarks.landmark[33]  # 왼쪽 눈 외측
        right_eye = face_landmarks.landmark[263]  # 오른쪽 눈 외측

        image_width, image_height = pil_image.size

        # 실제 픽셀 위치로 변환
        left_eye_pos = np.array([left_eye.x * image_width, left_eye.y * image_height])
        right_eye_pos = np.array([right_eye.x * image_width, right_eye.y * image_height])

        # 두 눈 사이의 각도 계산
        dx = right_eye_pos[0] - left_eye_pos[0]
        dy = right_eye_pos[1] - left_eye_pos[1]
        angle = math.degrees(math.atan2(dy, dx))  # 시계 방향이 양수

        # PIL은 반시계방향이 양수이므로 음수로 회전
        rotated_image = pil_image.rotate(-angle, resample=Image.BICUBIC, expand=True)

        return rotated_image

# === 기본 SSIM 비교 함수 (좌우 반전 포함, 이미지 객체 사용) ===
def compare_ssim_flipped_images(img1: Image.Image, img2: Image.Image) -> float:
    img1 = img1.convert("L")
    img2 = img2.convert("L")

    # 좌우 반전
    img1 = ImageOps.mirror(img1)

    # 크기 맞추기
    if img1.size != img2.size:
        img2 = img2.resize(img1.size)

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    score, _ = ssim(arr1, arr2, full=True)
    return round(score * 100, 2)

def compare_split_match(image: Image.Image) -> tuple[float, Image.Image, Image.Image]:
    width, height = image.size
    mid = width // 2

    # 좌/우 반으로 나누기
    left_half = image.crop((0, 0, mid, height))
    right_half = image.crop((mid, 0, width, height))

    # 오른쪽 반을 좌우 반전시켜 비교
    right_half_flipped = ImageOps.mirror(right_half)

    # 크기 맞추기
    if left_half.size != right_half_flipped.size:
        right_half_flipped = right_half_flipped.resize(left_half.size)

    # SSIM 계산
    arr1 = np.array(left_half.convert("L"))
    arr2 = np.array(right_half_flipped.convert("L"))
    score, _ = ssim(arr1, arr2, full=True)

    return round(score * 100, 2), left_half, right_half

# 가중치에 따른 평균 도출
def weighted_average(score_dict, weights):
        total_weighted_score = 0.0
        total_weight = 0.0
        for part, score in score_dict.items():
            if score is not None:
                weight = weights.get(part, 1.0)
                total_weighted_score += score * weight
                total_weight += weight
        return round(total_weighted_score / total_weight, 2) if total_weight else None

# 일치율 계산 함수
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
        del parts["nose"]  # 원본 nose 이미지 삭제

    if "mouth" in parts:
        score, left_mouth, right_mouth = compare_split_match(parts["mouth"])
        results["mouth"] = score
        parts["left_mouth"] = left_mouth
        parts["right_mouth"] = right_mouth
        del parts["mouth"]  # 원본 mouth 이미지 삭제

    results["chin"] = (
        compare_ssim_flipped_images(parts["left_chin"], parts["right_chin"])
        if "left_chin" in parts and "right_chin" in parts else None
    )

    return results
