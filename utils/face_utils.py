# utils/face_utils.py

from typing import List, Tuple

def estimate_position(landmarks: List[Tuple[float, float]], indices: List[int]) -> Tuple[int, int]:
    """
    지정된 인덱스의 랜드마크 좌표들의 평균 위치를 계산하여 반환합니다.
    - landmarks: (x, y) 튜플의 리스트
    - indices: 평균을 낼 랜드마크 인덱스 리스트
    """
    pts = [landmarks[i] for i in indices if i < len(landmarks)]
    if not pts:
        return (0, 0)
    avg_x = sum(x for x, y in pts) // len(pts)
    avg_y = sum(y for x, y in pts) // len(pts)
    return (avg_x, avg_y)
