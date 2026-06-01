# 팀 공유 폴더

팀원에게 전달할 수집 관련 파일은 이 폴더에서 관리합니다.

- `data_collection_bundle/`: 각 팀원이 내려받아 독립적으로 실행하는 데이터 수집 프로그램
- `../../PROJECT_GUIDE.md`: 전체 프로젝트 운영, 실행, 통합 절차
- `../docs/GEMINI_HANDOFF_PROMPT.md`: Gemini CLI에서 개발 작업을 이어갈 때 투입할 프롬프트

배포할 때는 `data_collection_bundle/`를 ZIP으로 압축해 공유합니다. 팀원이 반환하는
파일은 번들 내부의 `dataset/` 전체를 압축한 `dataset_<collector>_<YYYYMMDD>.zip`입니다.
