# 팀 공유 폴더

팀원에게 전달할 파일은 이 폴더에서 관리합니다.

- `PROJECT_GUIDE.md`: 전체 자모 분담, 양손 제어 규칙, Google Drive 업로드 및 통합 절차
- `data_collection_bundle/`: 각 팀원이 내려받아 독립적으로 실행하는 데이터 수집 프로그램
- `GEMINI_HANDOFF_PROMPT.md`: Gemini CLI에서 개발 작업을 이어갈 때 투입할 프롬프트

메인 프로젝트 개발 이력과 현재 작업 메모는 루트의 `sign_language_guide.md`에서
관리합니다.

배포할 때는 `data_collection_bundle/`를 ZIP으로 압축해 공유합니다. 수집기는
목표 프레임에 도달하면 저장을 멈추고 다음 담당 라벨을 자동 선택합니다. 수집 완료 후
팀원이 반환하는 파일은 번들 내부의 `dataset/` 전체를 압축한
`dataset_<collector>_<YYYYMMDD>.zip`입니다.
