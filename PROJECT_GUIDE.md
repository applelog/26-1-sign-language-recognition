# 정적 수화 인식 MVP 프로젝트 가이드

## 목표와 현재 범위

웹캠에서 정적 수화 손모양을 인식하고, SVM 모델로 자음과 모음을 분류한 뒤 한글
텍스트로 조합해 실시간 자막처럼 보여준다. 현재 1차 배포 범위는 정적 자모 입력,
삭제, 양손 시작/종료 제어, 방 코드 기반 영상통화형 웹 UI, WebSocket 자막 공유다.
동적 수화, 클라우드 배포, 실제 P2P 영상 송수신은 아직 범위에 포함하지 않는다.

## 프로젝트 구조

```text
AI_Project/
├── README.md
├── PROJECT_GUIDE.md
├── requirements.txt
├── backend/               # FastAPI 백엔드
├── frontend/              # 웹 UI
├── AI/
│   ├── train_model.py
│   ├── real_time_recognition.py
│   ├── evaluate_webcam.py
│   ├── gesture_config.json
│   ├── dataset/
│   ├── docs/              # 인수인계 프롬프트와 참고 자료
│   ├── models/
│   ├── reports/
│   ├── src/sign_language/
│   └── team_share/
└── trash/                 # 삭제 대신 보관한 파일
```

핵심 파일:

| 경로 | 역할 |
| --- | --- |
| `backend/main.py` | FastAPI API, 모델 로딩, 입력 상태 관리 |
| `frontend/` | MediaPipe Hands 기반 영상통화형 웹 UI |
| `AI/src/sign_language/angle_calculator.py` | 83차원 특징 추출 |
| `AI/src/sign_language/simple_svm_model.py` | scikit-learn SVM 학습/예측 |
| `AI/src/sign_language/runtime_state.py` | START/INPUT/RESULT 상태 관리 |
| `AI/src/sign_language/hangul_composer.py` | 자모 조합 |
| `AI/src/sign_language/control_gestures.py` | 양손 START/END 규칙 |
| `AI/gesture_config.json` | 라벨, 제어 라벨, 학습 대상 계약 |

## 모델과 입력 계약

- 오른손 입력은 MediaPipe 손 랜드마크에서 만든 83차원 특징을 사용한다.
- 특징 구성은 관절 각도 15개, 손목 기준 정규화 좌표 63개, 손끝 거리 5개다.
- 모델은 `StandardScaler + SVC(RBF, C=10, gamma=scale, class_weight=balanced)`이다.
- 학습 대상은 쌍자음 5개와 겹모음 4개를 제외한 자모 31개와 `DELETE(42)` 총 32개다.
- `START(40)`는 양손 펼친손, `END(41)`는 양손 주먹이며 현재 런타임에서는 규칙으로 판정한다.
- 웹 추론은 수집기와 좌표계를 맞추기 위해 landmark `x` 좌표를 미러링해서 백엔드에 보낸다.

제외/조합 규칙:

- `ㄲ`, `ㄸ`, `ㅃ`, `ㅆ`, `ㅉ`는 직접 수집하지 않고 기본 자음 반복으로 만든다.
- `ㅘ`, `ㅙ`, `ㅝ`, `ㅞ`는 직접 수집하지 않고 기본 모음 조합으로 만든다.
- `OTHER` 클래스는 사용하지 않는다.

입력 예시:

```text
ㅅ + ㅅ + ㅏ -> 싸
ㅇ + ㅗ + ㅏ -> 와
ㅇ + ㅜ + ㅔ -> 웨
ㅈ + ㅏ + ㄴ + ㅎ + ㅏ -> 잔하
ㅈ + ㅏ + ㄴ + ㅎ + ㅎ + ㅏ -> 잖하
```

## 실행 방법

의존성 설치:

```bash
python3 -m pip install -r requirements.txt
```

모델 학습:

```bash
python3 AI/train_model.py
```

웹 서버 실행:

```bash
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

브라우저:

```text
http://localhost:8000
```

같은 Wi-Fi 테스트:

```bash
python3.8 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

다른 기기 접속:

```text
http://<서버_IP>:8000
```

주의: `http://192.168.x.x:8000`처럼 IP로 접속하면 WebSocket 방 입장은 가능하지만
브라우저가 카메라 권한을 차단할 수 있다. 카메라는 보통 `HTTPS` 또는 `localhost`에서만
허용된다. 같은 Wi-Fi의 다른 기기에서 카메라까지 테스트하려면 로컬 HTTPS로 실행하거나
브라우저 개발 설정에서 해당 주소를 안전한 출처로 허용해야 한다.

서버 종료:

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
kill <PID>
```

또는 서버를 실행한 터미널에서 `Ctrl + C`를 누른다.

## 웹 사용법

1. 닉네임을 입력한다.
2. 역할을 선택한다.
3. `방 생성`을 누르거나 전달받은 방 코드를 입력하고 `방 입장`을 누른다.
4. 같은 방에 들어온 사용자는 WebSocket으로 입퇴장 상태와 자막을 공유한다.

수어 사용자:

1. 양손 펼친손으로 `START`를 입력한다.
2. 입력 모드가 되면 오른손으로 자음 또는 모음을 입력한다.
3. 같은 손모양을 잠깐 유지하면 글자가 확정된다.
4. 잘못 입력하면 오른손 `DELETE` 자세로 마지막 자모를 삭제한다.
5. 양손 주먹으로 `END`를 입력하면 입력이 종료된다.

비수어 사용자:

1. 비수어 사용자 역할로 입장한다.
2. 수어 인식 UI 대신 STT 중심 화면을 본다.
3. 마이크로 말하면 음성 자막이 자막 로그와 같은 방 사용자에게 전달된다.
4. 상대방 수어 자막은 자막 로그와 오른쪽 상단 상태 패널에 표시된다.

UI 표시:

- 왼쪽 상단 큰 텍스트는 현재 조합 결과다.
- 왼쪽 상단 토큰은 입력된 원시 자모와 확정 당시 confidence를 보여준다.
- 하단 카드는 현재 예측, confidence, top-3 후보, 디버그 상태를 보여준다.
- 오른쪽/하단 패널은 같은 방 사용자 상태와 수어/음성 자막 로그를 보여준다.
- 비수어 사용자 화면에서는 수어 토큰과 `/api/predict` 추론을 끄고 STT 상태를 보여준다.
- confidence 색상 기준은 `80% 이상 초록`, `50~79% 노랑`, `50% 미만 빨강`이다.

주의:

- 현재 버전은 실제 영상통화 연결이 아니라 영상통화형 UI다.
- 상대방에게는 카메라 영상이 아니라 WebSocket 자막, 역할, 상태 메시지만 전달된다.
- 오른쪽 상단 상대방 화면은 실제 영상이 아니라 상대방 이름, 역할, 최근 자막을 보여주는 상태 패널이다.
- 음성 자막(STT)은 브라우저 지원 여부에 따라 동작하지 않을 수 있다.
- 다른 기기가 IP 주소로 접속할 때 카메라가 안 뜨면 백엔드 문제가 아니라 브라우저
  보안 출처 제한일 가능성이 높다.

자주 보는 디버그 사유:

```text
right_hand_required       오른손으로 인식되지 않음
input_mode_not_started    START 전이라 입력 모드가 아님
stable_threshold_not_met  같은 손모양 유지가 부족함
no_hand                   손이 감지되지 않음
```

## 데이터셋 구조

프로젝트 통합 후 학습 데이터는 라벨 기준 폴더에 둔다. 수집자 정보는 파일명 prefix로 보존한다.

```text
AI/dataset/features/
├── label_00_ㄱ/gangwoo__label_0_ㄱ__....csv
├── label_03_ㄷ/heetae__label_3_ㄷ__....csv
├── label_19_ㅏ/heetae__label_19_ㅏ__....csv
└── label_42_DELETE/gangwoo__label_42_DELETE__....csv
```

팀원 수집 번들은 독립 실행을 위해 `dataset/features/<collector>/...csv` 형태로 저장한다.
통합 담당자가 프로젝트 본체에 병합할 때 라벨 기준 구조로 정리한다.

## 팀 데이터 수집 운영

팀원에게는 `AI/team_share/data_collection_bundle/`를 ZIP으로 전달한다. 팀원은 압축을 푼 뒤
자기 컴퓨터에서 실행한다.

```bash
python3 -m pip install -r requirements.txt
python3 web_server.py
```

브라우저에서 `http://localhost:8000`을 열고 본인 이름을 선택한다.

수집 규칙:

- 오른손 자모와 `DELETE`는 실제 오른손으로 수집한다.
- `START`/`END`는 강우가 양손으로 수집하며 `control_features/`에 별도 저장된다.
- `수집 시작` 후 5초 카운트다운이 끝나야 저장이 시작된다.
- 목표 프레임 `180`과 저장 간격 `220ms`는 고정이다.
- 목표 프레임에 도달하면 현재 세션 저장은 자동 종료되고 다음 담당 라벨이 선택된다.
- 다음 라벨 저장은 사용자가 다시 `수집 시작`을 눌러야 시작된다.
- 권장량은 라벨별 총 540프레임 이상이다.

담당 분배:

| 담당자 | 수집 라벨 |
| --- | --- |
| 강우 (`gangwoo`) | ㄱ, ㅈ, ㅌ, ㅐ, ㅔ, ㅜ, ㅠ, `START`, `END`, `DELETE` |
| 영래 (`youngrae`) | ㄹ, ㅅ, ㅍ, ㅑ, ㅕ, ㅡ |
| 정효 (`junghyo`) | ㄴ, ㅁ, ㅊ, ㅎ, ㅒ, ㅖ, ㅚ, ㅢ |
| 희태 (`heetae`) | ㄷ, ㅂ, ㅇ, ㅋ, ㅏ, ㅓ, ㅗ, ㅛ, ㅟ, ㅣ |

수집 완료 후 팀원은 번들 내부의 `dataset/` 전체를 압축해 Google Drive에 올린다.

```text
dataset_<collector>_<YYYYMMDD>.zip
```

## 평가와 검증

모델 학습 후 생성 파일:

```text
AI/models/sign_language_model.pkl
AI/reports/metrics/latest_summary.txt
AI/reports/metrics/classification_report.csv
AI/reports/metrics/confusion_matrix.csv
```

현재 데이터 기준 지표:

```text
Accuracy: 0.9980
Macro F1-score: 0.9980
Mean confidence: 0.9805
```

테스트:

```bash
python3 -m pytest AI/tests backend/tests
```

현재 로컬 기본 `python3`에서 `pytest`가 없으면 먼저 의존성을 설치하거나 pytest가
설치된 Python 환경을 사용한다.

2026-05-31 검증 결과:

```text
python3.8 -m pytest AI/tests backend/tests
23 passed in 2.35s
```

실제 웹캠 라벨 평가:

```bash
python3 AI/evaluate_webcam.py --label 0 --samples 50
```

예시 라벨:

```text
0  = ㄱ
19 = ㅏ
42 = DELETE
```

## 팀 역할

| 이름 | 역할 |
| --- | --- |
| 김강우 | 팀장, 전체 관리, AI 모델 개발, 데이터 관리, QA |
| 최희태 | 백엔드 개발 |
| 김정효 | 프론트엔드 개발 |
| 김영래 | 백엔드 개발 |

## 1차 배포 체크리스트

- [ ] `python3 AI/train_model.py`로 모델이 생성된다.
- [ ] `python3 -m pytest AI/tests backend/tests`가 통과한다.
- [ ] 웹에서 `START -> 자모 입력 -> DELETE -> END` 흐름이 동작한다.
- [ ] 방 생성/방 입장 후 같은 방 브라우저 간 자막 메시지가 전달된다.
- [ ] 나가기 후 재입장할 때 이전 WebSocket 연결이 남지 않는다.
- [ ] confidence 색상 기준이 UI에 반영된다.
- [ ] 팀원이 `AI/team_share/data_collection_bundle/`만 받아도 수집기를 실행할 수 있다.
- [ ] Google Drive 업로드 파일명이 `dataset_<collector>_<YYYYMMDD>.zip` 형식이다.

## 개발 로그

### 2026-05-28

- 팀 데이터 병합 후 오른손 32클래스 SVM 모델을 학습했다.
- 학습 데이터 구조를 수집자 폴더에서 라벨 기준 폴더로 정리했다.
- FastAPI 백엔드와 웹 UI를 연결했다.
- 웹 추론 좌표계를 수집기와 맞추기 위해 landmark `x` 좌표 미러링을 적용했다.
- top-3 예측 후보와 디버그 상태를 웹 UI에 표시했다.
- 조합 결과를 왼쪽 상단에 표시하고, confidence 기준 색상을 적용했다.
- `80% 이상 초록`, `50~79% 노랑`, `50% 미만 빨강` 기준을 UI 토큰, 후보, 카드 테두리에 적용했다.

### 2026-05-31

- `backend_AI_Project`를 1차 배포 기준 작업본으로 정했다.
- 영상통화형 방 UI와 WebSocket 자막 공유 흐름을 문서화했다.
- WebSocket 닉네임/방 코드 인코딩, 나가기 시 소켓 종료, 끊긴 소켓 broadcast 예외 처리를 보강했다.
