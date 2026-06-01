# Gemini CLI 인수인계 프롬프트

아래 내용을 Gemini CLI에 붙여 넣고 후속 작업을 진행한다.

```text
작업 디렉터리는 `/Users/kimtaemin/Desktop/G_AIproject/main_AI_project/backend_AI_Project`이다. 이 프로젝트는
웹캠에서 정적 수화 자모를 인식하고 한글 텍스트로 조합하는 FastAPI/Web/OpenCV/SVM MVP다.
동적 수화와 클라우드 배포는 현재 범위가 아니다. 변경 전에 관련 파일을 읽고,
83차원 특징과 라벨 ID 계약을 임의로 바꾸지 말라.

현재 구현:
- `backend/main.py`: FastAPI 백엔드, 모델 로딩, `/api/predict`, `/ws/{room_code}/{nickname}`
- `frontend/`: MediaPipe Hands 기반 영상통화형 웹 UI
- `AI/src/sign_language/angle_calculator.py`: 83차원 특징(각도 15 + 정규화 좌표 63 + 손끝 거리 5)
- `AI/train_model.py`, `AI/src/sign_language/simple_svm_model.py`: StandardScaler + RBF SVC 학습/예측
- `AI/src/sign_language/runtime_state.py`, `AI/src/sign_language/hangul_composer.py`: 입력 상태 및 한글 조합
- `AI/src/sign_language/control_gestures.py`, `AI/real_time_recognition.py`: 양손 제어 규칙과 OpenCV 실시간 표시
- `AI/models/sign_language_model.pkl`: 전체 학습 모델 산출물
- `AI/team_share/data_collection_bundle/`: 팀원 로컬 웹/CLI 수집 번들

현재 계약:
- 현재 오른손 SVM 학습 대상은 쌍자음 5개와 겹모음 `ㅘ(28)`, `ㅙ(29)`,
  `ㅝ(33)`, `ㅞ(34)`를 제외한 자모 31개와 오른손 `DELETE(42)` 총 32클래스다.
- 쌍자음 `ㄲ`, `ㄸ`, `ㅃ`, `ㅆ`, `ㅉ`는 별도 손모양을 수집하지 않고
  기본 자음 반복 입력으로 조합한다.
- 겹모음 `ㅘ`, `ㅙ`, `ㅝ`, `ㅞ`는 라벨 번호만 유지하고 수집/학습하지 않는다.
  입력은 `ㅗ+ㅏ`, `ㅗ+ㅐ`, `ㅜ+ㅓ`, `ㅜ+ㅔ` 기본 모음 조합으로 처리한다.
- `START(40)`는 양손 펼친손, `END(41)`는 양손 주먹이며 강우가
  `dataset/control_features/`에 별도로 수집한다. 현재 런타임에서는 규칙으로 판정한다.
- `OTHER` 라벨은 제거되었고 수집하거나 학습하지 않는다.
- 웹 추론은 수집기 좌표계와 맞추기 위해 frontend에서 landmark `x` 좌표를 미러링해 백엔드에 보낸다.
- 웹 UI confidence 색상 기준은 `80% 이상 초록`, `50~79% 노랑`, `50% 미만 빨강`이다.
- 현재 웹 UI는 실제 P2P 영상통화가 아니라 영상통화형 화면이다. 같은 방 사용자에게는
  WebSocket으로 입퇴장 상태, 수어 자막, 음성 자막만 전달한다.
- 방 코드와 닉네임은 frontend에서 URL 인코딩해서 WebSocket에 연결한다.
- 웹 수집기는 거울 입력의 handedness를 실제 손 기준으로 보정하고, 오른손
  수집 라벨은 실제 오른손, `START`/`END`는 양손이 감지되었을 때만 저장한다.
- 목표 프레임 `180`과 저장 간격 `220ms`는 UI에서 변경할 수 없는 고정값이다.
- `수집 시작`을 누르면 5초 카운트다운 후에만 실제 저장이 시작된다.
- 목표 프레임에 도달하면 저장을 끝내고 다음 담당 라벨을 자동 선택하되, 다음 세션 저장은 사용자가 시작한다.

팀 배정:
- gangwoo: ㄱ ㅈ ㅌ ㅐ ㅔ ㅜ ㅠ START END DELETE
- youngrae: ㄹ ㅅ ㅍ ㅑ ㅕ ㅡ
- junghyo: ㄴ ㅁ ㅊ ㅎ ㅒ ㅖ ㅚ ㅢ
- heetae: ㄷ ㅂ ㅇ ㅋ ㅏ ㅓ ㅗ ㅛ ㅟ ㅣ

운영 방식:
- 팀원은 각자 `team_share/data_collection_bundle`를 받아 localhost에서 수집한다.
- 결과물은 `dataset_<collector>_<YYYYMMDD>.zip`으로 Google Drive에 올린다.
- 통합 후 프로젝트 본체 학습 데이터는 `AI/dataset/features/label_XX_글자/collector__...csv` 구조다.
- 전체 데이터 전 강우 파일럿은 `python3 AI/train_model.py --pilot --collector gangwoo`와
  `python3 AI/real_time_recognition.py --pilot`을 사용하며, 최소 2개 학습 라벨 데이터가 필요하다.
- 2026-05-28 기준 양손 자모 추론 계획은 폐기했고, 겹모음은 기본 모음 조합으로 처리한다.

우선 확인 파일:
`AI/gesture_config.json`, `AI/team_share/data_collection_bundle/gesture_config.json`,
`PROJECT_GUIDE.md`, `backend/main.py`, `frontend/script.js`, `AI/real_time_recognition.py`.

검증:
`python3 -m json.tool AI/gesture_config.json`
`python3 -m json.tool AI/team_share/data_collection_bundle/gesture_config.json`
`python3 AI/train_model.py`
`python3 -m pytest AI/tests backend/tests`
`python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`

웹 수동 검증:
- 닉네임 입력 후 방 생성/방 입장
- 같은 방을 두 브라우저 탭에서 열고 자막/입퇴장 상태 전달 확인
- START -> 자모 입력 -> DELETE -> END
- 나가기 후 재입장 시 이전 WebSocket 연결이 남지 않는지 확인

팀 데이터 파일은 삭제하거나 덮어쓰지 말고, 변경 결과와 확인하지 못한 장치/데이터
의존 검증을 작업 종료 시 명확히 보고하라.
```
