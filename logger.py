import logging
import os
from datetime import datetime

# 로그 디렉토리 생성
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 오늘 날짜 기반 파일명 생성
today = datetime.now().strftime("%Y-%m-%d")
log_file = os.path.join(log_dir, f"app_{today}.log")

# 로그 포맷 설정
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 루트 로거 설정
logger = logging.getLogger("FAIcial")
logger.setLevel(logging.DEBUG)

# 중복 핸들러 방지
if not logger.handlers:
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # 핸들러 등록
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
