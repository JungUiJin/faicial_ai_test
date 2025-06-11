import os
import requests
import math
from datetime import datetime
from math import hypot
from PIL import Image, ImageDraw, ImageFont, ImageOps
from logger import logger
from utils.face_utils import estimate_position

# 폰트 경로 설정 (고정 크기)
FONT_URL = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTF/Korean/NotoSansKR-Regular.ttf"
FONT_PATH = os.path.join("fonts", "NotoSansKR-Regular.ttf")
if not os.path.exists(FONT_PATH):
    os.makedirs(os.path.dirname(FONT_PATH), exist_ok=True)
    resp = requests.get(FONT_URL)
    with open(FONT_PATH, "wb") as f:
        f.write(resp.content)
    logger.info("폰트 다운로드 완료: NotoSansKR-Regular.ttf")

def draw_dotted_line(draw, start, end, color="blue", width=2, dash_length=10):
    total = hypot(end[0] - start[0], end[1] - start[1])
    num = int(total // dash_length)
    if num < 1:
        draw.line([start, end], fill=color, width=width)
        return
    dx = (end[0] - start[0]) / num
    dy = (end[1] - start[1]) / num
    for i in range(0, num, 2):
        x1 = start[0] + dx * i
        y1 = start[1] + dy * i
        x2 = start[0] + dx * (i + 1)
        y2 = start[1] + dy * (i + 1)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

def project_point_to_line(px, py, pt1, pt2):
    ux, uy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    vx, vy = px - pt1[0], py - pt1[1]
    denom = ux*ux + uy*uy
    if denom == 0:
        return pt1
    t = (ux*vx + uy*vy) / denom
    return (pt1[0] + t*ux, pt1[1] + t*uy)

def crop_to_face_center_with_zoom(
    image: Image.Image,
    landmarks,
    h_ratio: float = 0.5,
    v_ratio: float = 4/5,
    min_face_occupancy: float = 0.6
):
    orig_w, orig_h = image.size

    # 얼굴 가로 중심 (귀끝 중간)
    lx, _ = landmarks[234]
    rx, _ = landmarks[454]
    face_cx = (lx + rx) / 2

    # 얼굴 세로 중심 및 높이 (머리·턱 중간)
    _, ty = landmarks[10]
    _, by = landmarks[152]
    face_cy = (ty + by) / 2
    face_h = by - ty

    # 4:5 비율 크롭 크기 결정
    ratio = 4 / 5
    if orig_w / orig_h >= ratio:
        crop_h = orig_h
        crop_w = int(crop_h * ratio)
    else:
        crop_w = orig_w
        crop_h = int(crop_w / ratio)

    # 얼굴 위치를 맞추기 위한 확대율 계산
    needed = [
        (crop_w * h_ratio) / face_cx,
        (crop_w * (1 - h_ratio)) / (orig_w - face_cx),
        (crop_h * v_ratio) / face_cy,
        (crop_h * (1 - v_ratio)) / (orig_h - face_cy),
    ]

    # 얼굴 세로 점유율(min_face_occupancy) 확보를 위한 확대율
    if face_h > 0:
        occ_scale = (min_face_occupancy * crop_h) / face_h
        needed.append(occ_scale)

    scale = max(1.0, *needed)
    scale = min(scale, 1.25)  # 최대 1.25배 확대 제한

    # 이미지 및 랜드마크 확대
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    landmarks = [(x * scale, y * scale) for x, y in landmarks]
    face_cx *= scale
    face_cy *= scale

    crop_w = min(crop_w, new_w)
    crop_h = min(crop_h, new_h)

    left = max(0, min(int(face_cx - crop_w * h_ratio), new_w - crop_w))
    top  = max(0, min(int(face_cy - crop_h * v_ratio), new_h - crop_h))

    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    new_landmarks = [(x - left, y - top) for x, y in landmarks]
    return cropped, new_landmarks

def generate_result_image(image: Image.Image, landmarks, score, part_scores):
    logger.debug("결과 이미지 시각화 시작")

    # 1) 얼굴 4:5 비율 확대 & 크롭
    image, landmarks = crop_to_face_center_with_zoom(
        image, landmarks,
        h_ratio=0.5,
        v_ratio=6/9,
        min_face_occupancy=0.5
    )

    # 2) 고정 해상도 리사이즈
    STANDARD_W, STANDARD_H = 800, 1000
    scale_img = STANDARD_W / image.width
    image = image.resize((STANDARD_W, STANDARD_H), Image.LANCZOS)
    landmarks = [(x * scale_img, y * scale_img) for x, y in landmarks]

    # 3) 해상도 기반 폰트 크기 동적 조절
    scale_factor = image.width / 800
    title_size = int(40 * scale_factor)
    message_size = int(34 * scale_factor)
    label_size = int(24 * scale_factor)
    face_size = int(15 * scale_factor)

    # 4) 폰트 설정
    font_title = ImageFont.truetype(FONT_PATH, title_size)
    font_message = ImageFont.truetype(FONT_PATH, message_size)
    font_label = ImageFont.truetype(FONT_PATH, label_size)
    font_face = ImageFont.truetype(FONT_PATH, int(face_size * 1.5))

    # 5) RGBA 모드
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    w, h = image.size

    # 6) 눈 좌표로 얼굴 대칭축(수직선) 벡터 계산
    x1, y1 = landmarks[33]   # 왼쪽 눈 외곽
    x2, y2 = landmarks[263]  # 오른쪽 눈 외곽
    ex, ey = x2 - x1, y2 - y1
    norm = math.hypot(ex, ey) or 1
    ux, uy = -ey / norm, ex / norm  # 수직 단위벡터

    # 7) 기준선 점(pt1, pt2) 계산
    L = max(w, h) * 2
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pt1 = (cx - ux * L, cy - uy * L)
    pt2 = (cx + ux * L, cy + uy * L)

    # 8) 기준선 그리기
    draw = ImageDraw.Draw(image)
    draw.line([pt1, pt2], fill='yellow', width=2)

    # 9) 상단 메시지 박스
    if score >= 90:
        message = "~(^ w ^~) 이 정도면 대칭의 신이에요! (~ ^ w ^)~"
    elif score >= 75:
        message = "☆대칭 미모의 숨겨진 고수~!☆"
    elif score >= 60:
        message = "살~짝 삐뚤, 그게 매력이라구요! ^^b"
    else:
        message = "비대칭? 그건 개성이라고 불러요 :)"

    image_center_x = w // 2
    vertical_padding = int(20 * scale_factor)
    box_height = int(title_size * 3 + vertical_padding * 2)
    start_y = vertical_padding

    overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([20, start_y, w - 20, start_y + box_height], fill=(0, 0, 0, 180))
    image = Image.alpha_composite(image, overlay)
    draw = ImageDraw.Draw(image)

    def safe_text(draw_obj, text, x, y, font, fill, anchor='mm'):
        bbox = draw_obj.textbbox((x, y), text, font=font, anchor=anchor)
        left, top, right, bottom = bbox
        dx = dy = 0
        if top < 0:
            dy = -top + 5
        elif bottom > h:
            dy = h - bottom - 5
        if left < 0:
            dx = -left + 5
        elif right > w:
            dx = w - right - 5
        draw_obj.text((x + dx + 1, y + dy + 1), text, font=font, fill='black', anchor=anchor)
        draw_obj.text((x + dx, y + dy), text, font=font, fill=fill, anchor=anchor)

    safe_text(draw, f'당신의 대칭률은 {score:.2f}%!!', image_center_x, start_y + vertical_padding + title_size * 0.5, font_title, 'white')
    safe_text(draw, message, image_center_x, start_y + vertical_padding + title_size * 2.5, font_message, 'white')

    # 10) 거리 시각화 (기울어진 대칭축에 대한 최단 거리)
    highlights = [
        (61,  'blue', 'left_mouth'), (291, 'blue', 'right_mouth'),
        (133, 'blue', 'left_eye'),   (362, 'blue', 'right_eye'),
        (234, 'cyan', 'left_ear'),   (454, 'cyan', 'right_ear'),
        (98,  'red', 'left_nose'),   (327, 'red', 'right_nose'),
        (172, 'red', 'left_chin'),   (397, 'red', 'right_chin'),
    ]
    distance_dict = {}
    for idx, color, name in highlights:
        x_i, y_i = landmarks[idx]
        proj = project_point_to_line(x_i, y_i, pt1, pt2)
        draw_dotted_line(draw, (x_i, y_i), proj, color=color)

        text_x = int((x_i + proj[0]) / 2)
        text_y = int((y_i + proj[1]) / 2)
        safe_text(draw, f"{int(hypot(x_i - proj[0], y_i - proj[1]))}px",
                  text_x, text_y, font_face, color)

        distance_dict[name] = round(hypot(x_i - proj[0], y_i - proj[1]), 0)

    # 11) 부위별 라벨
    LABEL_W, LABEL_H = 150, 50
    PADDING = 20
    label_indices = {'눈': 33, '코': 1, '입': 13, '귀': 234, '턱': 397}
    static_pos = {}
    for part, idx in label_indices.items():
        x_pt, y_pt = landmarks[idx]
        bx = PADDING if part in ['눈', '입'] else w - LABEL_W - PADDING
        by = int(y_pt - LABEL_H / 2)
        by = max(PADDING, min(by, h - LABEL_H - PADDING))
        static_pos[part] = (bx, by)

    shadow_offset = 2
    shadow_color = (0, 0, 0, 100)
    key_map = {'눈': 'eyes', '코': 'nose', '입': 'mouth', '귀': 'ears', '턱': 'chin'}

    for part, (bx, by) in static_pos.items():
        txt = f"{part}: {part_scores.get(key_map[part], 0):.1f}%"
        draw.rounded_rectangle(
            [bx + shadow_offset, by + shadow_offset, bx + LABEL_W + shadow_offset, by + LABEL_H + shadow_offset],
            fill=shadow_color, radius=8)
        draw.rounded_rectangle([bx, by, bx + LABEL_W, by + LABEL_H], fill='white', radius=8)
        safe_text(draw, txt, bx + LABEL_W // 2, by + LABEL_H // 2, font_label, 'black')

    return image, distance_dict
