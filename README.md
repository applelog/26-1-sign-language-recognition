# 정적 수화 인식 실시간 자막 서비스

웹캠으로 정적 수화 손모양을 인식하고, SVM 모델로 자음/모음을 분류한 뒤
한글 텍스트로 조합해 실시간 자막처럼 보여주는 MVP입니다.

현재 범위는 정적 수화입니다. 동적 수화는 정적 인식 흐름이 안정화된 뒤 확장합니다.

## 주요 기능

- 브라우저 또는 OpenCV에서 MediaPipe Hands 랜드마크 추출
- SVM 모델 기반 오른손 정적 자모/DELETE 분류
- 양손 `START`, `END` 제어 자세로 입력 세션 관리
- 쌍자음은 기본 자음 반복 입력으로 조합
- `ㅘ`, `ㅙ`, `ㅝ`, `ㅞ`는 기본 모음 조합으로 생성
- FastAPI 백엔드와 웹 UI 연동
- 방 코드 기반 영상통화형 UI와 WebSocket 자막 공유
- `수어 사용자` / `비수어 사용자` 역할 선택
- 비수어 사용자용 STT 중심 화면
- Accuracy, Macro F1-score, Confusion Matrix, 라벨별 Precision/Recall/F1 리포트 생성

## 프로젝트 구조

```text
26-1-sign-language-recognition/
├── AI/                    # 모델, 학습/평가/실시간 실행 코드
├── backend/               # FastAPI 백엔드
├── frontend/              # 웹 UI
├── README.md              # GitHub용 대표 문서
├── LICENSE                # 소스코드 라이선스(Apache-2.0)
├── DATA_LICENSE.md        # 데이터/모델 라이선스(CC BY-NC 4.0)
├── PROJECT_GUIDE.md       # 팀원/개발자용 상세 운영 문서
├── PROJECT_PRESENTATION_NOTES.md
├── codex.md               # 작업 인수인계 문서
└── requirements.txt       # 공통 Python 의존성
```

## 설치

```bash
python -m pip install -r requirements.txt
```

## 모델 사용 및 재학습

GitHub에는 바로 실행 가능한 학습 완료 모델을 포함합니다.

```text
AI/models/sign_language_model.pkl
```

전체 원본 학습 데이터는 용량과 관리 문제 때문에 GitHub에 포함하지 않습니다.
재학습이 필요하면 Google Drive에서 팀 데이터셋을 내려받아 `AI/dataset/features/`에 배치한 뒤 실행합니다.

```bash
python AI/train_model.py
```

학습/평가 결과:

- 모델: `AI/models/sign_language_model.pkl`
- 성능 요약: `AI/reports/metrics/latest_summary.txt`
- 라벨별 지표: `AI/reports/metrics/classification_report.csv`
- Confusion Matrix: `AI/reports/metrics/confusion_matrix.csv`

학습 데이터는 라벨 기준 폴더로 관리합니다. 수집자 정보는 파일명 prefix로 남깁니다.

```text
AI/dataset/features/
├── label_00_ㄱ/gangwoo__label_0_ㄱ__....csv
├── label_03_ㄷ/heetae__label_3_ㄷ__....csv
└── label_19_ㅏ/heetae__label_19_ㅏ__....csv
```

현재 학습 데이터 기준 주요 지표:

- Accuracy: `0.9980`
- Macro F1-score: `0.9980`
- Mean confidence: `0.9805`

## 웹 실행

```bash
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

브라우저에서 접속:

```text
http://localhost:8000
```

같은 Wi-Fi에서 다른 기기도 접속해야 하면 서버를 전체 네트워크에 열어야 합니다.

```bash
python3.8 -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

다른 기기에서는 서버 컴퓨터의 IP로 접속합니다.

```text
http://<서버_IP>:8000
```

단, 브라우저 카메라는 `HTTPS` 또는 `localhost`에서만 안정적으로 허용됩니다.
다른 기기가 `http://192.168.x.x:8000`으로 접속하면 방 입장과 자막 공유는 되지만
카메라가 차단될 수 있습니다. 실제 팀 테스트에서는 로컬 HTTPS로 실행하거나
브라우저 개발 설정에서 해당 주소를 안전한 출처로 허용해야 합니다.

웹 흐름:

```text
닉네임 입력
-> 역할 선택(수어 사용자 또는 비수어 사용자)
-> 방 생성 또는 방 코드 입장
-> 같은 방 사용자와 자막 상태 공유
-> 양손 펼침 START
-> 오른손 자모 입력
-> 필요 시 오른손 DELETE
-> 양손 주먹 END
-> 조합 텍스트 출력
```

비수어 사용자 흐름:

```text
닉네임 입력
-> 비수어 사용자 선택
-> 방 생성 또는 방 코드 입장
-> 마이크 STT로 음성 자막 전송
-> 상대방 수어 자막 수신
```

수어 입력 흐름:

```text
양손 펼침 START
-> 오른손 자모 입력
-> 필요 시 오른손 DELETE
-> 양손 주먹 END
-> 조합 텍스트 출력
```

자세한 실행, 종료, 디버그 확인 방법은 `PROJECT_GUIDE.md`를 참고하세요.

## OpenCV 실행

```bash
python AI/real_time_recognition.py
```

## 실제 웹캠 평가

예: `ㄱ(label 0)`을 50프레임 평가

```bash
python AI/evaluate_webcam.py --label 0 --samples 50
```

결과는 `AI/reports/metrics/webcam_eval_<timestamp>.json`에 저장됩니다.

## API 요약

- `GET /api/health`: 모델/서버 상태 확인
- `GET /api/config`: 라벨, 제어 라벨, feature dimension 조회
- `POST /api/predict`: 손 랜드마크 기반 예측 및 텍스트 조합
- `POST /api/reset`: 입력 세션 초기화
- `WS /ws/{room_code}/{nickname}`: 같은 방 사용자 간 수어/음성 자막과 입퇴장 상태 공유

프론트엔드는 영상 원본을 보내지 않고 MediaPipe Hands의 21개 랜드마크와
handedness만 백엔드로 전송합니다. 웹 추론은 수집기와 좌표계를 맞추기 위해
landmark `x` 좌표를 미러링한 뒤 전송하며, 응답에는 디버깅용 top-3 예측 후보가
포함됩니다.

현재 웹 UI는 실제 P2P 영상통화가 아니라, 영상통화처럼 보이는 단일 로컬 카메라
화면과 자막 공유 UI입니다. 같은 방의 다른 브라우저에는 WebSocket으로 자막과
상태 메시지만 전달합니다.

오른쪽 상단 상대방 패널은 실제 카메라 영상이 아니라 상대방 이름, 역할, 접속 상태,
최근 자막을 보여주는 상태 패널입니다. 실제 상대방 카메라 영상 송수신은 아직 구현하지
않았습니다.

## 팀 역할

| 이름 | 역할 |
| --- | --- |
| 김강우 | 팀장, 전체 관리, AI 모델 개발, 데이터 관리, QA |
| 최희태 | 백엔드 개발 |
| 김정효 | 프론트엔드 개발 |
| 김영래 | 백엔드 개발 |

## 참고

- 팀 공유/수집 번들은 `AI/team_share/`에 보관합니다.
- 상세 프로젝트 문서는 `PROJECT_GUIDE.md`, 작업 규칙은 `codex.md`에 보관합니다.
- AI/CLI 인수인계 문서는 `AI/docs/`에 보관합니다.
- `trash/`는 삭제 대신 이동해 둔 파일 보관 폴더입니다.

## License

이 저장소는 코드와 데이터/모델의 라이선스를 분리합니다.

- Source code: Apache License 2.0. 자세한 내용은 `LICENSE`를 확인하세요.
- Dataset, trained model, and data-derived artifacts: Creative Commons
  Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). 자세한 내용은
  `DATA_LICENSE.md`를 확인하세요.

`AI/models/sign_language_model.pkl`, GitHub 또는 Google Drive로 공유되는
`AI/dataset/**`, `AI/team_share/**`의 수집 데이터와 산출물, 성능 리포트 등
학습 데이터 기반 산출물은 비상업적 용도로만 사용할 수 있습니다. 상업적 이용,
재판매, 유료 서비스 통합, 상업적 재배포는 프로젝트 팀의 별도 서면 허가가
필요합니다.
