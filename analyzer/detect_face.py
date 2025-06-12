import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from logger import logger

mp_face_mesh = mp.solutions.face_mesh

def detect_landmarks(image_bytes: bytes):
    # 이미지 바이트 → OpenCV 이미지
    logger.debug("이미지 바이트 수신 및 디코딩 시도")
    image_array = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        logger.error("이미지 디코딩 실패: 유효하지 않은 이미지")
        raise ValueError("Invalid image data")

    logger.debug("OpenCV 이미지 디코딩 성공")

    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # MediaPipe 모델 초기화
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,           # 정적 이미지 처리
        max_num_faces=1,                  # 최대 얼굴 수: 1
        refine_landmarks=True,            # 눈, 입술 등 세부 랜드마크 보정
        min_detection_confidence=0.5      # 감지 신뢰도 임계값
    ) as face_mesh:

        results = face_mesh.process(image_rgb)

        # 얼굴이 감지되지 않음
        if not results.multi_face_landmarks:
            logger.warning("얼굴이 감지되지 않음")
            return None, None

        logger.debug("얼굴 랜드마크 감지 성공")

        # 첫 번째 얼굴의 랜드마크 추출
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image_rgb.shape
        landmarks = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks.append((x, y))

        # OpenCV 이미지를 PIL 이미지로 변환
        image_pil = Image.fromarray(image_rgb)

        return landmarks, image_pil


def align_and_detect_landmarks(image_bytes: bytes):
    # 이미지 바이트 → OpenCV 이미지
    logger.debug("이미지 바이트 수신 및 디코딩 시도")
    image_array = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        logger.error("이미지 디코딩 실패: 유효하지 않은 이미지")
        raise ValueError("Invalid image data")

    logger.debug("OpenCV 이미지 디코딩 성공")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:

        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            logger.warning("얼굴이 감지되지 않음")
            return None, None

        logger.debug("얼굴 랜드마크 감지 성공")
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = image_rgb.shape

        # 눈 좌표 추출 (좌: 33, 우: 263)
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        left_eye_pos = np.array([left_eye.x * w, left_eye.y * h])
        right_eye_pos = np.array([right_eye.x * w, right_eye.y * h])

        # 회전 각도 계산 (눈 중심을 수평으로 정렬)
        delta = right_eye_pos - left_eye_pos
        angle = np.degrees(np.arctan2(delta[1], delta[0]))

        logger.debug(f"얼굴 회전 각도: {angle:.2f}도")

        # 회전 행렬 계산 및 이미지 회전
        center = tuple(np.array([w // 2, h // 2]))
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned_image = cv2.warpAffine(image_rgb, rot_mat, (w, h), flags=cv2.INTER_LINEAR)

        # 회전된 이미지로 다시 랜드마크 감지
        results_aligned = face_mesh.process(aligned_image)

        if not results_aligned.multi_face_landmarks:
            logger.warning("얼굴이 회전된 이미지에서도 감지되지 않음")
            return None, None

        aligned_landmarks = []
        for lm in results_aligned.multi_face_landmarks[0].landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            aligned_landmarks.append((x, y))

        aligned_pil_image = Image.fromarray(aligned_image)

        return aligned_landmarks, aligned_pil_image