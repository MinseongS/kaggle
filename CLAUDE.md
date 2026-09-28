# Kaggle monorepo

목표: Kaggle Competitions Master (금 1 + 은 2). **솔로 참가만** 한다 (팀 병합 제안 금지).

## 구조
- `competitions/<slug>/` — 대회별 **독립 uv 프로젝트** (각자 pyproject.toml / uv.lock). uv workspace로 묶지 않는다 (대회마다 torch 등 버전이 다름).
- `libs/kkit/` — 공용 유틸. 대회 프로젝트에서 path(editable) 의존성으로 사용. 두 대회 이상에서 반복된 코드만 여기로 올린다.
- `insights/` — 대회를 넘어서는 지식.
  - `common.md`: 대회 종류와 무관한 원칙
  - `playbooks/<type>.md`: 유형별 노하우 (tabular, cv, nlp-llm, agent-sim, science)
  - `solutions/<slug>.md`: 과거/종료 대회 상위 솔루션 요약
  - `postmortems/<slug>.md`: 내가 참가한 대회 회고
- `templates/competition/` — 새 대회 템플릿. `scripts/new-comp.sh <slug>` 로 생성.

## 규칙
- 데이터·모델·출력물은 git에 넣지 않는다 (`data/`, `outputs/`, `models/`).
- 모든 실험은 `EXPERIMENTS.md`에 한 줄 기록: 날짜 | exp id | 변경점 | CV | LB | 메모. **CV를 LB보다 신뢰**한다.
- 대회 `README.md` 상단에 평가지표, 마감일, 제출 형식(코드 대회 여부, 런타임 제한), 현재 best CV/LB를 유지한다.
- 대회 종료 후 반드시 `insights/postmortems/`에 회고를 쓰고, 상위 솔루션을 `insights/solutions/`에 요약하고, 일반화 가능한 교훈은 `common.md`/playbook에 반영한다.
- Kaggle 작업(다운로드/제출/커널 push)은 `kaggle` CLI, 디스커션·노트북·리더보드 탐색 등 CLI로 안 되는 건 ego-browser.
- Python은 uv로만 관리 (`uv add`, `uv run`). 코드 대회는 Kaggle 이미지의 라이브러리 버전에 맞춘다.
- C/C++ 가 필요하면 해당 대회 폴더의 `cpp/`에 둔다 (numba로 충분한지 먼저 검토).
