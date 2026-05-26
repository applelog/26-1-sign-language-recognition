# Gemini CLI 인수인계 프롬프트

아래 내용을 Gemini CLI에 붙여 넣고 후속 작업을 진행한다.

```text
작업 디렉터리는 `/Users/kimtaemin/Desktop/G_AIproject/AI_Project`이다. 이 프로젝트는
웹캠에서 정적 수화 자모를 인식하고 한글 텍스트로 조합하는 OpenCV/SVM MVP다.
동적 수화와 클라우드 배포는 현재 범위가 아니다. 변경 전에 관련 파일을 읽고,
83차원 특징과 라벨 ID 계약을 임의로 바꾸지 말라.

현재 구현:
- `angle_calculator.py`: 83차원 특징(각도 15 + 정규화 좌표 63 + 손끝 거리 5)
- `train_model.py`, `simple_svm_model.py`: StandardScaler + RBF SVC 학습/예측
- `runtime_state.py`, `hangul_composer.py`: 입력 상태 및 한글 조합
- `control_gestures.py`, `real_time_recognition.py`: 양손 제어 규칙과 OpenCV 실시간 표시
- `team_share/data_collection_bundle/`: 팀원 로컬 웹/CLI 수집 번들

현재 계약:
- SVM 학습 대상은 자모 ID `0..39` 전체와 오른손 `DELETE(42)` 총 41클래스다.
- `START(40)`는 양손 펼친손, `END(41)`는 양손 주먹이며 강우가
  `dataset/control_features/`에 별도로 수집한다. 현재 런타임에서는 규칙으로 판정한다.
- `OTHER` 라벨은 제거되었고 수집하거나 학습하지 않는다.
- 웹 수집기는 실제 오른손만 저장하도록 거울 입력의 handedness를 보정했다.
- 목표 프레임 `180`과 저장 간격 `220ms`는 UI에서 변경할 수 없는 고정값이다.
- `수집 시작`을 누르면 5초 카운트다운 후에만 실제 저장이 시작된다.
- 목표 프레임에 도달하면 저장을 끝내고 다음 담당 라벨을 자동 선택하되, 다음 세션 저장은 사용자가 시작한다.

팀 배정:
- gangwoo: ㄱ ㄸ ㅃ ㅈ ㅌ ㅐ ㅔ ㅘ ㅜ ㅠ START END DELETE
- youngrae: ㄲ ㄹ ㅅ ㅉ ㅍ ㅑ ㅕ ㅙ ㅝ ㅡ
- junghyo: ㄴ ㅁ ㅆ ㅊ ㅎ ㅒ ㅖ ㅚ ㅞ ㅢ
- heetae: ㄷ ㅂ ㅇ ㅋ ㅏ ㅓ ㅗ ㅛ ㅟ ㅣ

운영 방식:
- 팀원은 각자 `team_share/data_collection_bundle`를 받아 localhost에서 수집한다.
- 결과물은 `dataset_<collector>_<YYYYMMDD>.zip`으로 Google Drive에 올린다.
- 통합 담당자는 루트 `dataset/`에 병합하고 모든 41클래스 및 83차원 형식을 검증한 뒤 학습한다.
- 전체 데이터 전 강우 파일럿은 `python train_model.py --pilot --collector gangwoo`와
  `python real_time_recognition.py --pilot`을 사용하며, 최소 2개 학습 라벨 데이터가 필요하다.
- 2026-05-26 기준 통합 데이터와 학습 모델이 없다면 실시간 성능을 단정하지 말라.

우선 확인 파일:
`gesture_config.json`, `team_share/data_collection_bundle/gesture_config.json`,
`team_share/PROJECT_GUIDE.md`, `sign_language_guide.md`, `real_time_recognition.py`.

검증:
`python3 -m json.tool gesture_config.json`
`python3 -m json.tool team_share/data_collection_bundle/gesture_config.json`
`python3 -m pytest -q -p no:cacheprovider`
`node --check team_share/data_collection_bundle/web_ui/app.js`

팀 데이터 파일은 삭제하거나 덮어쓰지 말고, 변경 결과와 확인하지 못한 장치/데이터
의존 검증을 작업 종료 시 명확히 보고하라.
```
