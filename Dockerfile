FROM python:3.10

# 필요한 시스템 패키지 설치 (OpenCV가 필요로 함)
RUN apt-get update && apt-get install -y libgl1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]