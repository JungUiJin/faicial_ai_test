# 1. 베이스 이미지 지정
FROM python:3.10-slim

# 2. OpenCV 등 실행에 필요한 시스템 라이브러리 설치
# - libglib2.0-0 추가 (일부 opencv 설치 시 필요)
# - libsm6, libxext6 등도 일부 환경에서 필요
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 앱 소스 복사
COPY . .

# 6. 앱 실행 명령어
CMD ["python", "app.py"]
