from flask import Flask, request, jsonify
from analyzer.detect_face import detect_landmarks, align_and_detect_landmarks
from analyzer.analyze_symmetry import calculate_symmetry
from analyzer.visualize_result import generate_result_image
from analyzer.image_devide import compare_match_parts_from_images, get_face_parts
from logger import logger
from utils.image_utils import encode_image_to_base64
from utils.visual_utils import draw_landmark_points, draw_specific_points
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["https://faicial.site"])  # 운영용: 정확한 출처만 허용

# 전역 호출 카운터
call_counters = {
    "debug_landmarks": 0,
    "analyze": 0
}

# ──────────────────────────────────────────────────────────────────────────────
# DEBUG LANDMARKS ENDPOINT
@app.route("/debug_landmarks", methods=["POST"])
def debug_landmarks():
    # 호출 횟수 증가 및 로그
    call_counters["debug_landmarks"] += 1
    logger.info(f"[debug_landmarks] 호출 횟수: {call_counters['debug_landmarks']}회")

    logger.info("디버그 랜드마크 요청 수신됨")
    if "image" not in request.files:
        logger.warning("요청에 이미지 파일 없음")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        landmarks, image = detect_landmarks(image_bytes)
        if landmarks is None:
            logger.warning("얼굴이 감지되지 않음")
            return jsonify({"error": "No face detected"}), 400

        logger.info(f"검출된 랜드마크 개수: {len(landmarks)}")
        debug_img = draw_landmark_points(image, landmarks, color="lime", radius=2)
        debug_img = draw_specific_points(debug_img, landmarks, [234, 454], color="red", radius=6)
        img_data = encode_image_to_base64(debug_img)

        logger.info("디버그 랜드마크 이미지 생성 및 전송 완료")
        return jsonify({"image_base64": img_data})

    except Exception as e:
        logger.exception("디버그 랜드마크 처리 중 예외 발생")
        return jsonify({"error": str(e)}), 500

# ──────────────────────────────────────────────────────────────────────────────
# ANALYZE ENDPOINT
@app.route("/analyze", methods=["POST"])
def analyze():
    # 호출 횟수 증가 및 로그
    call_counters["analyze"] += 1
    logger.info(f"[analyze] 호출 횟수: {call_counters['analyze']}회")

    logger.info("분석 요청 수신됨 v5")
    if "image" not in request.files:
        logger.warning("요청에 이미지 파일 없음")
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    try:
        logger.debug("얼굴 랜드마크 추출 시도")
        landmarks, image = detect_landmarks(image_bytes)
        if landmarks is None:
            logger.warning("얼굴이 감지되지 않음")
            return jsonify({"error": "No face detected"}), 400

        logger.debug(f"랜드마크 수: {len(landmarks)}")

        align_landmarks, align_image = align_and_detect_landmarks(image_bytes)
        logger.debug("대칭률 계산 시작")
        symmetry_score, part_scores = calculate_symmetry(align_landmarks)
        logger.debug(f"총 대칭률 점수: {symmetry_score}")
        logger.debug(f"부위별 대칭률 점수: {part_scores}")

        logger.debug("일치율 계산 시작")
        parts_images = get_face_parts(align_landmarks, align_image)
        match_scores = compare_match_parts_from_images(parts_images)
        logger.debug(f"부위별 일치율 : {match_scores}")

        encoded_parts = {
            part_name: encode_image_to_base64(part_image)
            for part_name, part_image in parts_images.items()
        }

        weights = {
            "eyes": 0.30,
            "nose": 0.20,
            "mouth": 0.20,
            "chin": 0.20,
            "ears": 0.10
        }

        final_scores = {}
        weighted_total = 0.0
        for part, weight in weights.items():
            match = match_scores.get(part, 0)
            if part == "chin":
                final = round(match, 2)
            else:
                sym = part_scores.get(part, 0)
                final = round((sym * 0.5 + match * 0.5), 2)
            final_scores[part] = final
            weighted_total += final * weight

        final_score = round(weighted_total, 2)
        logger.debug(f"일치율 + 대칭률 : {final_scores}")
        logger.debug(f"최종 대칭 점수 : {final_score}")

        result_image, distance_dict = generate_result_image(image, landmarks, final_score, final_scores)
        img_data = encode_image_to_base64(result_image)

        logger.info("분석 성공 및 응답 반환")
        logger.info("결과 이미지 Base64 생성 및 전송 완료")

        return jsonify({
            "parts_images": encoded_parts,
            "final_scores": final_scores,
            "final_score": final_score,
            "result_image": img_data,
            "total_distance": distance_dict
        })

    except Exception as e:
        logger.exception("분석 중 예외 발생")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Flask 앱 실행 시작")
    app.run(host='0.0.0.0', port=5000, debug=True)
