# 정적 수화 인식 MVP 팀 운영 가이드

## 목표와 범위

웹캠에서 정적 수화 손모양을 인식하고 자음과 모음을 한글 텍스트로 조합한다.
현재 범위는 오른손 정적 자모 입력과 삭제이며, 동적 수화는 포함하지 않는다.

## 현재 계약

- 특징값: MediaPipe 손 랜드마크 기반 83차원 벡터
- 학습 모델: `StandardScaler + SVC(RBF)`
- SVM 학습 대상: 자모 40개 `0..39`와 `DELETE(42)` 총 41개 클래스
- 입력 제어: `START(40)`는 양손 펼친손, `END(41)`는 양손 주먹
- 강우는 `START`와 `END`의 양손 특징도 수집한다. 이는 런타임 규칙 확인용
  `dataset/control_features/`에 별도 저장되며 단일 손 SVM 학습 CSV와 섞지 않는다.
- `DELETE(42)`는 오른손 엄지 아래 정적 자세로 수집하고 강우가 담당한다.
- 기존 `OTHER` 클래스는 제거했다. 일반 동작 오입력은 우선 양손 시작 게이트와 안정화 판정으로 줄인다.

## 담당 분배

| 담당자 | 수집 라벨 |
| --- | --- |
| 강우 (`gangwoo`) | ㄱ, ㄸ, ㅃ, ㅈ, ㅌ, ㅐ, ㅔ, ㅘ, ㅜ, ㅠ, `START`, `END`, `DELETE` |
| 영래 (`youngrae`) | ㄲ, ㄹ, ㅅ, ㅉ, ㅍ, ㅑ, ㅕ, ㅙ, ㅝ, ㅡ |
| 정효 (`junghyo`) | ㄴ, ㅁ, ㅆ, ㅊ, ㅎ, ㅒ, ㅖ, ㅚ, ㅞ, ㅢ |
| 희태 (`heetae`) | ㄷ, ㅂ, ㅇ, ㅋ, ㅏ, ㅓ, ㅗ, ㅛ, ㅟ, ㅣ |

## 수집 절차

팀원에게는 `team_share/data_collection_bundle/`를 압축한 ZIP만 전달한다.
각 팀원은 압축을 푼 폴더에서 실행한다.

```bash
python -m pip install -r requirements.txt
python web_server.py
```

브라우저에서 `http://localhost:8000`을 열고 자신의 이름을 선택한다. 화면에는
자신에게 배정된 수집 라벨만 표시된다.

- 자모와 `DELETE`는 실제 오른손을, `START`와 `END`는 양손을 카메라에 보이게 한다.
- `수집 시작` 버튼을 누른 뒤 5초 준비 카운트다운이 끝나야 CSV 저장이 시작된다.
- 목표 프레임 수 `180`과 저장 간격 `220ms`는 팀 공통 고정값이며 화면에서 수정할 수 없다.
- 목표 수량에 도달하면 현재 세션 저장은 자동 종료되고 더 이상 해당 자세를 저장하지 않는다.
- 완료 직후 작업 큐는 다음 담당 라벨을 자동 선택한다. 자세 변경 후 사용자가 수집 시작을 누른다.
- 권장 확보량은 라벨별 합계 `180프레임 x 3세션 = 540프레임` 이상이다.
- 세션마다 손 위치, 거리, 방향, 조명을 조금씩 바꾼다.
- 잘못된 포즈가 들어간 세션은 Google Drive에 올리지 않고 다시 수집한다.

## 데이터 반환

수집 완료 후 번들 안의 `dataset/` 폴더 전체를 압축해 Google Drive에 올린다.

```text
dataset_<collector>_<YYYYMMDD>.zip
└── dataset/
    ├── features/<collector>/...csv
    ├── control_features/<collector>/...csv
    └── sessions/...json
```

영상 원본, `.DS_Store`, `__pycache__/`, `*.pyc`, `.pytest_cache/`는 포함하지 않는다.

## 통합과 실행

통합 담당자는 반환 ZIP의 `features/`와 `sessions/`를 프로젝트 루트의
`dataset/`에 병합한다. 모든 학습 대상 ID와 83개 특징 열이 존재하는지 확인한 뒤
실행한다.

```bash
python train_model.py
python real_time_recognition.py
```

시연 흐름은 `양손 펼침 START -> 오른손 자모 입력 -> 오른손 DELETE 필요 시 입력
-> 양손 주먹 END`이다. 실시간 입력은 안정화된 예측만 반영한다.

전체 데이터가 모이기 전 강우 데이터로 수집/분류 흐름만 시험할 때는 부분 모델을
별도 파일로 만든다. 자모 또는 `DELETE` 학습 라벨이 최소 2개 이상 수집되어야 한다.
`--pilot`은 팀 공유 번들 안에 방금 저장된 `dataset/features/`도 직접 읽는다.

```bash
python train_model.py --pilot --collector gangwoo
python real_time_recognition.py --pilot
```

`pilot_sign_language_model.pkl`은 임시 확인용이며 최종 모델 성능을 의미하지 않는다.

## 체크리스트

- [ ] 각 팀원 화면에 본인 담당 라벨만 나타나는지 확인했다.
- [ ] 실제 오른손을 보여 줄 때 오른손으로 표시되는지 확인했다.
- [ ] 목표 프레임 도달 후 저장이 멈추고 다음 담당 라벨이 선택되는지 확인했다.
- [ ] 자모 40개와 `DELETE`의 학습 데이터가 모두 반환되었다.
- [ ] 강우가 `START`/`END` 양손 데이터를 `control_features/`로 반환했다.
- [ ] 병합 데이터로 SVM 학습 및 OpenCV 입력 흐름을 확인했다.
