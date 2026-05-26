# 정적 수화 인식 프로젝트 가이드

## 목표

웹캠에서 정적 수화 손모양을 분류하고, 인식한 자음과 모음을 조합해 실시간
텍스트로 표시하는 MVP를 만든다. 목적은 수어 사용자와 비수어 사용자 사이의
소통 보조이며, 동적 수화는 정적 인식이 안정화된 뒤 검토한다.

## 현재 구조

| 경로 | 역할 |
| --- | --- |
| `angle_calculator.py` | MediaPipe 손 랜드마크에서 83차원 특징 추출 |
| `simple_svm_model.py`, `train_model.py` | SVM 학습과 모델 저장 |
| `hangul_composer.py`, `runtime_state.py` | 자모 조합과 입력 상태 처리 |
| `control_gestures.py` | 양손 `START`/`END` 규칙 판정 |
| `real_time_recognition.py` | OpenCV 실시간 인식 및 자막 표시 |
| `gesture_config.json` | 학습 대상, 라벨 ID, 제어 규칙의 기준 설정 |
| `team_share/data_collection_bundle/` | 팀원별 로컬 데이터 수집 번들 |
| `team_share/PROJECT_GUIDE.md` | 배정, 수집, 반환, 통합 절차 |
| `team_share/GEMINI_HANDOFF_PROMPT.md` | 다른 CLI 작업 인계 프롬프트 |

## 학습 및 입력 계약

- 손 특징은 관절 각도 15개, 손목 기준 정규화 좌표 63개, 손끝 거리 5개의 총
  83개 수치를 사용한다.
- 모델은 `StandardScaler + SVC(RBF, C=10, gamma=scale, class_weight=balanced)`이다.
- SVM 학습 대상은 자모 전체 ID `0..39`와 `DELETE(42)` 총 41개이다.
- `DELETE`는 오른손 엄지 아래 정적 자세이며 학습 대상으로 수집한다.
- `START(40)`는 양손 펼친손, `END(41)`는 양손 주먹으로 판정한다. 강우가 두
  제어 동작의 양손 특징도 별도 수집하며, 현재 단일 손 SVM 학습에는 포함하지 않는다.
- 이전에 검토했던 `OTHER` 일반 손동작 클래스는 제거했다. 현재는 양손 시작
  게이트와 최근 프레임 안정화 판정으로 임의 입력을 제한한다.

실시간 흐름:

```text
양손 펼친손 START -> 오른손 자모/DELETE 입력 -> 양손 주먹 END -> 결과 표시
```

## 데이터 수집 운영

팀원은 `team_share/data_collection_bundle/`를 각자 컴퓨터에서 실행한다.

```bash
cd team_share/data_collection_bundle
python -m pip install -r requirements.txt
python web_server.py
```

브라우저는 `http://localhost:8000`을 사용한다. 웹 수집기는 자동 분류기가
아니므로 선택한 라벨과 실제 자세가 일치하는지 수집자가 직접 확인한다.

- 화면은 미러 보기지만 handedness는 실제 오른손 기준으로 보정되어 있다.
- 담당자를 선택하면 그 사람의 학습 대상 작업만 큐에 표시된다.
- `수집 시작` 후 5초 카운트다운이 지나야 프레임 저장이 시작된다.
- 목표 프레임 `180`과 저장 간격 `220ms`는 고정이며 화면에서 수정할 수 없다.
- 목표 프레임에 도달하면 저장은 멈추고 작업 큐가 다음 담당 라벨을 선택한다.
  사용자는 다음 자세를 준비한 뒤 새 수집 세션을 직접 시작한다.
- 라벨별 권장량은 총 `180프레임 x 최소 3세션`이며, 짧은 세션을 여러 번
  모으는 것도 가능하다.

담당 분배:

| 담당자 | 자모/제어 수집 |
| --- | --- |
| 강우 | ㄱ, ㄸ, ㅃ, ㅈ, ㅌ, ㅐ, ㅔ, ㅘ, ㅜ, ㅠ, `START`, `END`, `DELETE` |
| 영래 | ㄲ, ㄹ, ㅅ, ㅉ, ㅍ, ㅑ, ㅕ, ㅙ, ㅝ, ㅡ |
| 정효 | ㄴ, ㅁ, ㅆ, ㅊ, ㅎ, ㅒ, ㅖ, ㅚ, ㅞ, ㅢ |
| 희태 | ㄷ, ㅂ, ㅇ, ㅋ, ㅏ, ㅓ, ㅗ, ㅛ, ㅟ, ㅣ |

강우의 `START`와 `END` CSV는 `dataset/control_features/`에 저장되며 양손
규칙 확인용이다. 자모/`DELETE` SVM 파일은 `dataset/features/`에 저장된다.

수집 완료 후 각 팀원은 번들의 `dataset/` 전체를
`dataset_<collector>_<YYYYMMDD>.zip`으로 압축해 Google Drive에 올린다.
통합 담당자는 루트 `dataset/features/`, `dataset/sessions/`에 병합한다.

## 학습과 실행

```bash
python train_model.py
python real_time_recognition.py
```

`train_model.py`는 기본적으로 `active_label_ids`의 41개 학습 대상이 모두 있을
때 모델을 저장한다. 현재 평가 결과는 프레임 단위 랜덤 분할이므로 사람이나
세션이 바뀐 상황의 일반화 성능으로 해석하지 않는다.

강우 데이터만으로 동작을 먼저 점검할 때는 학습 라벨을 최소 2개 이상 수집한
뒤 임시 모델을 사용한다. `--pilot` 실행은 공유 번들에서 수집한
`dataset/features/`도 별도 병합 없이 읽는다.

```bash
python train_model.py --pilot --collector gangwoo
python real_time_recognition.py --pilot
```

이 실행은 `pilot_sign_language_model.pkl`을 사용하며 수집된 라벨만 예측한다.

## 현재 상태와 다음 검증

- 2026-05-26 기준 학습 계약은 전체 자모 + `DELETE`로 전환되었다.
- 팀 데이터가 병합되기 전에는 완성 모델의 실제 인식률을 검증할 수 없다.
- 데이터 병합 후에는 라벨별 수량, 83차원 형식, 혼동 행렬, 실제 웹캠 흐름을
  순서대로 점검한다.

## 개발 로그

### 2026-03-30

- 각도 기반 정적 수화 인식 프로젝트 골격과 수집/학습/실시간 실행 초안을 작성했다.

### 2026-04-27

- 브라우저 기반 로컬 데이터 수집 방식을 도입하고 팀원별 CSV 병합 운영을 정리했다.

### 2026-05-18

- 기본 모델을 scikit-learn SVM 파이프라인으로 전환했다.

### 2026-05-26

- 초성/중성 조합, 삭제, 입력 상태와 팀 공유 번들 구조를 추가했다.
- 초기 일부 클래스 수집 계획을 폐기하고 자모 전체 `0..39`와 `DELETE(42)`를
  학습하는 팀 분할 방식으로 변경했다.
- `START`는 양손 펼친손, `END`는 양손 주먹의 규칙 판정으로 변경하고
  `OTHER` 클래스는 제거했다.
- 웹 수집기의 실제 오른손 판정 문제를 보정하고, 담당자 작업 큐 및 목표 프레임
  도달 시 저장 종료 후 다음 라벨을 선택하는 흐름으로 바꿨다.
- 강우가 `START`와 `END` 양손 특징을 별도 제어 데이터로 수집하도록 복구하고,
  수집 수량과 저장 간격을 팀 공통 고정값으로 잠갔다.
- 수집 시작 전 5초 준비 시간과 부분 라벨 파일럿 모델 학습/실행 옵션을 추가했다.
