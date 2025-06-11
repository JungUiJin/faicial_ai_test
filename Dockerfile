FROM python:3.10

# OpenCV 실행에 필요한 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]