# 프론트엔드 변경 요약

## 변경 파일

- `frontend/index.html`
- `frontend/style.css`
- `frontend/script.js`

## 추가된 기능

### 1. 입장 화면 추가

- 서비스 첫 화면에 닉네임 입력 UI를 추가했습니다.
- 방 코드 입력창을 추가했습니다.
- `방 생성`, `방 입장` 버튼을 추가했습니다.
- 방 생성 시 임시 방 코드가 자동 생성됩니다.

### 2. 카메라 권한 요청 흐름 변경

- 기존에는 페이지 접속 즉시 카메라가 시작되었습니다.
- 변경 후에는 방 생성 또는 방 입장 시점에 카메라 권한을 요청합니다.
- 카메라 권한 확인에 실패하면 안내 후 통화 화면으로 이동하며, 카메라 오류 상태를 표시합니다.
- 다른 기기가 `http://192.168.x.x:8000`으로 접속해 카메라가 차단되면 HTTPS/localhost
  보안 출처 문제를 안내합니다.

### 3. 통화 화면 구조 추가

- 기존 수어 인식 UI 디자인은 유지했습니다.
- 상단에 방 코드와 접속 상태를 표시합니다.
- `내 화면` 상태 표시 영역을 추가했습니다.
- `상대방 화면` 대기 패널을 추가했습니다.
- 마이크, 카메라, 나가기 버튼을 추가했습니다.

### 4. 나가기 및 이전 방 재입장

- 나가기 버튼을 누르면 입장 화면으로 돌아갑니다.
- 나가기 시 카메라 스트림을 종료하고 캔버스를 초기화합니다.
- 직전에 입장했던 닉네임과 방 코드를 유지합니다.
- `이전 방 다시 입장` 버튼으로 같은 방에 다시 들어갈 수 있습니다.

### 5. 자막 로그 영역 추가

- 수어/STT 결과를 표시할 자막 로그 영역을 추가했습니다.
- 수어 입력이 완료되면 자막 로그에 결과가 누적됩니다.
- 향후 STT 결과와 사용자 상태 메시지도 같은 영역에 표시할 수 있도록 구성했습니다.

### 6. 실시간 메시지 처리

- 백엔드 WebSocket 연동을 위해 `handleRealtimeMessage()` 함수를 추가했습니다.
- 방 코드와 닉네임은 URL 인코딩해서 WebSocket에 연결합니다.
- 나가기 시 카메라 스트림과 WebSocket 연결을 함께 종료합니다.
- 아래 형식의 메시지를 처리할 수 있습니다.

```js
{
  type: "sign_result",
  user: "사용자1",
  text: "안녕하세요",
  confidence: 0.98
}
```

```js
{
  type: "stt_result",
  user: "사용자2",
  text: "오늘 기분 어때요?"
}
```

```js
{
  type: "user_status",
  user: "사용자2",
  status: "joined"
}
```

## 유지한 부분

- 기존 VisionOS 스타일의 글래스 UI를 최대한 유지했습니다.
- 기존 웹캠 기반 수어 인식 흐름을 유지했습니다.
- 기존 `/api/predict`, `/api/reset` 연동을 유지했습니다.
- 기존 confidence 색상 기준을 유지했습니다.

## 테스트 결과

초기 프론트엔드 적용 당시 아래 검증을 통과했습니다.

```powershell
node --check frontend\script.js
```

```powershell
.\.venv\Scripts\python.exe -m pytest AI/tests backend/tests
```

테스트 결과:

```text
23 passed, 3 warnings
```

경고는 scikit-learn 모델 저장 버전 차이로 발생하며, 이번 프론트엔드 변경과는 무관합니다.

2026-05-31 backend 기준 추가 변경에서는 WebSocket 연결 안정화와 문서 최신화를 진행했습니다.
기본 `python3` 환경에는 FastAPI/pytest가 없어서 실패했지만, `python3.8` 환경에서
`python3.8 -m pytest AI/tests backend/tests`를 실행해 `23 passed`를 확인했습니다.

## 참고

- `frontend/backups/` 폴더는 작업 중 만든 백업본입니다.
- 실제 적용되는 파일은 `frontend/index.html`, `frontend/style.css`, `frontend/script.js`입니다.
