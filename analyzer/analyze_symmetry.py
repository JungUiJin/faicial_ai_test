# analyzer/analyze_symmetry.py

import numpy as np

# MediaPipe landmark index 기준 좌우 짝 (예: 좌: 33, 우: 263)
# 각 쌍은 (left_idx, right_idx)
# “nose” 파트를 추가하여 코 양 옆 팁(98, 327)도 대칭률에 포함합니다.
PAIR_INDICES = {
    "eyes": [(33, 263), (160, 387), (159, 386)],
    "mouth": [(61, 291), (78, 308), (95, 324)],
    "ears": [(234, 454), (172, 397), (152, 378)],
    "nose": [(98, 327)],
}

def calculate_symmetry(landmarks):
    if not landmarks or len(landmarks) < 468:
        raise ValueError("Insufficient landmark points.")

    # 중심선 기준 x좌표 (코 중심 기준)
    left_x = landmarks[234][0]
    right_x = landmarks[454][0]
    center_x = (left_x + right_x) / 2

    part_scores = {}
    total_diffs = []

    for part, pairs in PAIR_INDICES.items():
        diffs = []
        for left_idx, right_idx in pairs:
            lx, ly = landmarks[left_idx]
            rx, ry = landmarks[right_idx]

            # 좌우 점을 중심선 기준으로 반사시켜 차이 계산
            reflected_rx = 2 * center_x - rx
            diff = np.sqrt((lx - reflected_rx)**2 + (ly - ry)**2)
            diffs.append(diff)
            total_diffs.append(diff)

        # 부위 평균 → 정규화 (0~100점으로 변환)
        avg_diff = np.mean(diffs)
        part_score = max(0, 100 - avg_diff)
        part_scores[part] = round(part_score, 2)

    # 전체 평균
    overall_diff = np.mean(total_diffs)
    symmetry_score = round(max(0, 100 - overall_diff), 2)

    return symmetry_score, part_scores
