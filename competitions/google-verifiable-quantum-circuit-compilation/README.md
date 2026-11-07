# google-verifiable-quantum-circuit-compilation

https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation

| 항목 | 값 |
|---|---|
| 유형 | science / 최적화 (kickmix 회로 압축, 검증기 루프) |
| 평가지표 | 3과제 (ref 비용/내 비용)의 기하평균. invsqrt: Q×(Toffoli+T/4), dqi: R(ε+)×Q²×T, dfthcsos: λ_eff×T×Q |
| 마감 | 2027-01-20 23:59 UTC (엔트리·병합 01-13) |
| 제출 형식 | 파일 제출 (zip: invsqrt.kmb, dqi.kmb, dfthcsos/angle_k_*.kmb + submission.npz). 채점 ~45분. 5회/일, 최종 2개 |
| 메달 지급 | 예 (Featured, $150k) |
| 팀 수 / 메달 컷 (11/03 public) | 2162팀 / 금(~14위) ~26.0, 은(~108위) 24.43. 1위 28.83, 중앙값 23.14 |
| Best CV / LB | 24.94093 / 24.94093 (2026-11-06 19:10 auto-harvest, 11-07 08:31 기준 315/2507, 은 컷 24.96481). 주의: 11-07 07:14 제출은 로컬 24.94136 / LB 24.87108 (dqi 가 비공개 salt 에서 2샷 실패) (CV = 로컬 verifier, salt 없음) |

## 참고한 과거 솔루션
-

## 현재 계획
- 로컬 환경: PyPI `kickmix` 1.0.0 (macOS arm64 wheel 있음). GPU 불필요, CPU 코어가 탐색 속도 결정
- 1) sample_submission 을 로컬 verifier 로 검증해 1.0 재현 2) 최고 공개 노트북의 과제별 점수 분해 3) 개선 여지 큰 과제에 에이전트 탐색 집중
