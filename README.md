# 🧠 FAIcial_AI

🎭 FAIcial 프로젝트: 얼굴 대칭 분석 어플리케이션

---

## 📌 프로젝트 개요

FAIcial_AI는 업로드된 얼굴 이미지에서 **MediaPipe** 기반 랜드마크를 추출하고, 좌우 대칭률을 계산하여 시각화 결과를 반환하는 **Flask** 서버 애플리케이션입니다.

- 클라이언트는 이미지 파일을 전송하면, 서버는 대칭 점수 및 부위별 점수와 함께 **PNG** 이미지를 Base64 문자열로 반환합니다.

---

## 🛠️ 사용 기술 스택

| 기술            | 설명                                 |
| --------------- | ------------------------------------ |
| **Python 3.12** | 메인 개발 언어                       |
| **Flask**       | 경량 웹 프레임워크 (API 서버)        |
| **MediaPipe**   | 얼굴 랜드마크 검출 (Google FaceMesh) |
| **OpenCV**      | 이미지 디코딩 및 전처리              |
| **Pillow**      | 이미지 처리 및 Base64 인코딩         |
| **NumPy**       | 수치 계산 및 분석 지원               |
| **Requests**    | 외부 폰트 다운로드                   |
| **Logging**     | 통합 로그 관리 (`logger.py`)         |

---

## 📁 프로젝트 폴더 구조

<details>
<summary>클릭하여 펼치기</summary>

```plaintext
FAIcial_AI/
 │
 ├── app.py                        # 🔹 Flask 엔트리 포인트
 ├── requirements.txt              # 🔹 의존성 목록
 ├── README.md                     # 🔹 전체 설명 문서
 ├── CHANGELOG.md                  # 🔹 개선 이력 정리
 ├── logger.py                     # 🔹 통합 로그 설정 모듈
 │
 ├── fonts/                        # 🔤 폰트 저장 경로
 │   └── NotoSansKR-Regular.otf
 │
 ├── analyzer/                     # 얼굴 분석 로직
 │   ├── __init__.py
 │   ├── detect_face.py            # 얼굴 인식 및 랜드마크 추출
 │   ├── analyze_symmetry.py       # 대칭률 계산 로직
 │   └── visualize_result.py       # 결과 이미지 시각화
 │
 ├── utils/                        # 유틸 함수 모듈
 │   ├── __init__.py               # 패키지 초기화
 │   ├── image_utils.py            # 이미지 Base64 인코딩 유틸
 │   └── face_utils.py             # 랜드마크 좌표 유틸
 │
 ├── test_images/                  # 🧪 테스트용 이미지 (Git 추적 제외)
 │   ├── sample1.jpg
 │   ├── sample2.png
 │   └── sample3.jpg
 │
 ├── outputs/                      # 💾 결과 이미지 저장 폴더 (Git 추적 제외)
 │   └── result_YYYYMMDD_HHMMSS.png
 │
 └── logs/                         # 📝 로그 파일 저장 위치 (Git 추적 제외)
     └── app_YYYY-MM-DD.log
```

</details>

---

## ⚙️ 설치 방법

```bash
# 가상환경 생성 (선택 사항)
python -m venv venv
source venv/bin/activate  # 또는 venv\Scripts\activate (Windows)

# 의존성 설치
pip install -r requirements.txt
```

---

## 🚀 실행 방법

```bash
export FLASK_APP=app.py         # Windows PowerShell: $Env:FLASK_APP = "app.py"
export FLASK_ENV=development    # 개발 모드 활성화 (선택)
```

- **서버 실행**: `python app.py` 또는 `flask run`
- **기본 주소**: `http://127.0.0.1:5000`

---

## 🔌 API 요청/응답 예시 (`POST /analyze`)

### 요청

- **URL**: `http://127.0.0.1:5000/analyze`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Body**:

  | 필드명 | 타입 | 설명               |
  | ------ | ---- | ------------------ |
  | image  | File | 분석할 얼굴 이미지 |

### 응답 (200 OK)

```json
{
  "symmetry_score": 87.4,
  "part_symmetries": {
    "eyes": 93.5,
    "mouth": 91.9,
    "jaw": 76.7
  },
  "image_base64": "data:image/png;base64,iVBORw0KGgoAAA..."
}
```

---

## ✅ 전체 진행 체크리스트

- [x] Flask 서버 기본 엔드포인트(`/analyze`) 구현
- [x] MediaPipe FaceMesh 연동 및 랜드마크 추출
- [x] 대칭률 계산 로직 구현 (`analyze_symmetry.py`)
- [x] 결과 이미지 시각화 및 크롭/리사이즈(`visualize_result.py`)
- [x] Base64-only 응답 기능 구현 및 디스크 저장 로직 주석 처리
- [x] `utils/image_utils.py`로 Base64 인코딩 분리
- [x] `utils/face_utils.py`로 랜드마크 평균 위치 계산 분리
- [x] 테스트 및 Postman Visualizer 검증
- [ ] 단위 테스트 추가
- [ ] Docker 환경 설정
- [ ] CI/CD 파이프라인 구축

---

## 🧠 개발 흐름

```plaintext
1️⃣ 클라이언트 → Flask `/analyze` 요청 (multipart/form-data)
2️⃣ detect_face.py: MediaPipe FaceMesh로 얼굴 랜드마크 추출
3️⃣ analyze_symmetry.py: 좌우 대칭 차이 연산 → 전체/부위별 점수 계산
4️⃣ visualize_result.py: 얼굴 크롭 → 랜드마크·점수 오버레이 → PIL 이미지 반환
5️⃣ image_utils.py: PIL 이미지 → Base64 data URI 변환
6️⃣ Flask 응답: JSON 형태로 점수 및 Base64 이미지 반환
```

---

## ✨ 기타 주의사항

- **`.venv/`, `logs/`, `outputs/`, `test_images/`** 디렉터리는 Git에서 제외되어 있습니다.
- 외부 폰트 다운로드를 위해 네트워크 연결이 필요합니다.
- 개발 모드에서는 Flask 내장 서버를 사용하며, 프로덕션 환경에서는 WSGI 서버(e.g., Gunicorn) 사용을 권장합니다.

---

## 🏁 최종 목표

- **FAIcial 서비스**의 백엔드 핵심 모듈로서 안정적 얼굴 대칭 분석 API 제공
- 프론트엔드 및 모바일 앱에서 실시간 결과 렌더링 연동 지원
- 향후 Docker 및 CI/CD 통합으로 배포 자동화 구축
