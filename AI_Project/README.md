# 정적 수화 인식 실시간 자막 서비스

웹캠으로 정적 수화 손모양을 인식하고, 분류된 자음과 모음을 한글 텍스트로
조합하여 실시간 자막으로 보여주는 프로젝트입니다. 수어 사용자와 비수어
사용자 사이의 소통 장벽을 낮추는 것을 목표로 합니다.

현재는 정적 수화 기반 MVP를 개발하고 있으며, 향후 동적 수화 인식과 웹 기반
실시간 자막 서비스로 확장할 예정입니다.

## 주요 기능

- 웹캠 영상에서 MediaPipe Hands 손 랜드마크 추출
- 정적 수화 자음·모음과 삭제 동작을 SVM 모델로 분류
- `START`와 `END` 양손 제어 동작을 사용한 입력 세션 관리
- 인식된 자음과 모음의 한글 조합 및 결과 텍스트 표시
- 팀원별 브라우저 데이터 수집기와 통합 학습 흐름 제공
- OpenCV 실행기 및 웹 데모 UI를 통한 파일럿 확인

## 입력 흐름

```text
양손 펼친손 START
  -> 오른손 정적 자모 입력
  -> 필요 시 오른손 DELETE
  -> 양손 주먹 END
  -> 조합된 텍스트 출력
```

| ID | 입력 | 처리 방식 |
| --- | --- | --- |
| `0..18` | 자음 | 오른손 특징 기반 SVM 분류 |
| `19..39` | 모음 | 오른손 특징 기반 SVM 분류 |
| `40` | `START` | 양손 펼친손 규칙 판정 |
| `41` | `END` | 양손 주먹 규칙 판정 |
| `42` | `DELETE` | 오른손 특징 기반 SVM 분류 |

## 기술 구성

| 영역 | 기술 및 역할 |
| --- | --- |
| Hand Tracking | MediaPipe Hands |
| Feature Engineering | 관절 각도, 정규화 좌표, 손끝 거리 기반 83차원 특징 |
| Classification | scikit-learn `StandardScaler + SVC(RBF)` |
| Text Assembly | 자모 입력 상태 관리 및 한글 조합 로직 |
| Desktop Demo | OpenCV + Pillow 기반 실시간 인식 화면 |
| Web Collection | 브라우저 손 추적 기반 팀 데이터 수집기 |
| Web Demo Prototype | `gemini/` UI와 Python HTTP 추론 서버 |

현재 구현된 AI 모델은 MLP가 아닌 **SVM `.pkl` 모델**입니다. 웹 데모
프로토타입은 Python HTTP 서버가 모델을 로드해 `/api/predict`로 결과를
반환하는 형태이며, FastAPI/WebSocket 기반 서비스 구조는 후속 개발 범위입니다.

## 모델과 데이터

오른손 정적 입력은 MediaPipe 랜드마크에서 계산한 83개 특징값을 모델 입력으로
사용합니다.

```text
15개 관절 각도 + 63개 정규화 좌표 + 5개 손끝 거리 = 83개 특징
```

- 학습 대상 모델 파일: `sign_language_model.pkl`
- 파일럿 테스트 모델: `pilot_sign_language_model.pkl`
- 자모/`DELETE` 수집 데이터: `team_share/data_collection_bundle/dataset/features/`
- 양손 `START`/`END` 확인 데이터: `team_share/data_collection_bundle/dataset/control_features/`
- 라벨 기준 설정: `gesture_config.json`

`START`와 `END`는 현재 단일 손 SVM 학습에 포함하지 않고 양손 규칙으로
처리합니다.

## 실행 방법

### 1. 의존성 설치

```bash
python -m pip install -r requirements.txt
```

### 2. 데이터 수집기 실행

팀원별 데이터 수집은 독립 번들에서 진행합니다.

```bash
cd team_share/data_collection_bundle
python web_server.py
```

브라우저에서 `http://localhost:8000`에 접속합니다. 수집은 시작 버튼을 누른
후 5초 뒤 시작되며, 라벨별 목표 프레임이 채워지면 저장이 끝나고 다음 담당
라벨이 선택됩니다.

### 3. 파일럿 모델 학습 및 OpenCV 테스트

일부 강우 데이터만으로 동작을 확인할 때 사용합니다.

```bash
python train_model.py --pilot --collector gangwoo
python real_time_recognition.py --pilot
```

파일럿 모델은 수집된 일부 라벨만 구분하므로 최종 인식 성능을 의미하지
않습니다.

### 4. 전체 모델 학습 및 실행

팀 데이터가 모두 병합된 이후 실행합니다.

```bash
python train_model.py
python real_time_recognition.py
```

## 웹 서비스 연동 방향

현재 웹 데모 프로토타입은 브라우저에서 손 랜드마크를 추출한 뒤 Python
추론 서버로 전송하여 예측 결과와 조합 텍스트를 표시하는 흐름을 사용합니다.

```text
Webcam -> MediaPipe Hands -> Backend Inference -> SVM Prediction
       -> Stable Event Handling -> Hangul Assembly -> Subtitle UI
```

다음 단계에서는 백엔드를 FastAPI 기반으로 정리하고, 실시간 자막 표시를 위한
통신 인터페이스를 명확히 정의할 계획입니다.

## 프로젝트 구조

```text
AI_Project/
├── gesture_config.json                 # 라벨 및 제어 입력 계약
├── angle_calculator.py                 # 83차원 특징 추출
├── train_model.py                      # SVM 학습
├── real_time_recognition.py            # OpenCV 실시간 실행
├── runtime_state.py                    # 안정화 및 입력 상태 관리
├── hangul_composer.py                  # 한글 조합
├── gemini/                             # 웹 데모 UI 및 추론 서버 프로토타입
└── team_share/data_collection_bundle/  # 팀원용 데이터 수집 프로그램
```

데이터 수집 및 팀 운영 절차는
[team_share/PROJECT_GUIDE.md](team_share/PROJECT_GUIDE.md)에서 관리합니다.

## 팀 구성

| 이름 | 역할 |
| --- | --- |
| 김강우 | 팀장, 전체 관리, AI 모델 개발, 데이터 관리, QA |
| 최희태 | 백엔드 개발 |
| 김정효 | 프론트엔드 개발 |
| 김영래 | 백엔드 개발 |

## 개발 현황

- 정적 수화 데이터 수집기 구현
- 팀원별 라벨 분배 및 데이터 병합 규칙 수립
- SVM 파일럿 모델 학습 및 OpenCV 실시간 인식 검증
- 자모 조합, 삭제, 양손 시작·종료 제어 흐름 구현
- 웹 데모 UI와 Python 추론 서버 프로토타입 개발 진행

## Roadmap

- 전체 자음·모음 데이터 수집 및 최종 SVM 학습
- 웹 UI와 추론 백엔드 연동 안정화
- FastAPI 기반 실시간 자막 API 구조 정리
- 다양한 사용자 환경에서 인식 성능 평가
- 정적 수화 안정화 이후 동적 수화 인식 확장 검토
