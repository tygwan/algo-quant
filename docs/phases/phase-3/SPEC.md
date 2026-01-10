# Phase 3: Regime Classification

## 목표
FRED 거시경제 지표를 활용한 경기 사이클 분류

## 범위

### 포함
- FRED 지표 처리 (GDP, 실업률, 수익률곡선)
- 경기 사이클 분류 (확장, 수축, 침체, 회복)
- 체제 기반 시그널 생성

### 제외
- 실시간 체제 예측 모델
- 딥러닝 기반 분류

## 기술 상세

### 경기 사이클 정의
| 체제 | 특징 | 투자 전략 |
|------|------|-----------|
| Expansion | GDP↑, 실업률↓ | 주식 비중↑ |
| Contraction | GDP↓, 실업률↑ | 방어주, 채권 |
| Recession | NBER 공식 침체 | 현금, 안전자산 |
| Recovery | 침체 후 반등 | 경기민감주 |

### 주요 지표
```python
LEADING_INDICATORS = [
    "PERMIT",      # 건축 허가
    "AWHMAN",      # 제조업 근무시간
    "UMCSENT",     # 소비자 신뢰지수
]

COINCIDENT_INDICATORS = [
    "PAYEMS",      # 비농업 고용
    "INDPRO",      # 산업생산지수
    "DPCERAM1M",   # 개인소득
]

LAGGING_INDICATORS = [
    "UNRATE",      # 실업률
    "CPILFESL",    # 근원 인플레이션
]
```

## 완료 조건

- [ ] 지표 처리 파이프라인 구현
- [ ] 분류 알고리즘 검증
- [ ] 과거 NBER 사이클과 비교 검증
- [ ] 단위 테스트 80%+

## 예상 산출물

```
src/regime/
├── __init__.py
├── indicators.py
├── classifier.py
└── signals.py
```
