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
