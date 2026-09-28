# Agent / Simulation / 최적화 (ConnectX, Lux, Santa, ARC, 게임 AI 등)
- 평가가 LB 매칭(에이전트끼리 대결)이면 로컬 대결 환경 + Elo 측정부터 만든다
- 규칙 기반 heuristic → 탐색(MCTS, beam) → 학습(imitation / RL) 순으로 쌓는다
- 속도가 곧 점수: numba → C++(pybind11/nanobind) 순으로 최적화
- Santa류 조합 최적화: 좋은 초기해 + local search / SA. C++ solver가 사실상 표준
