# Phase 3: Regime Classification Specification

## 목표

FRED 거시경제 지표를 활용하여 시장 체제(Market Regime)를 분류하고, 체제에 따른 투자 시그널을 생성합니다.

## 범위

### 1. FRED 지표 처리기

FRED API에서 수집한 거시경제 지표를 가공하여 체제 분류에 사용할 수 있는 형태로 변환합니다.

**주요 지표**:
- GDP 성장률 (GDPC1)
- 실업률 (UNRATE)
- 인플레이션 (CPIAUCSL, PCE)
- 금리 (FEDFUNDS, DFF)
- 수익률 곡선 (T10Y2Y, T10Y3M)
- 산업생산지수 (INDPRO)
- 소비자심리지수 (UMCSENT)
- ISM 제조업 PMI (NAPM)
- 신규 실업수당 청구 (ICSA)

**처리 기능**:
- 시계열 정규화 (z-score)
- 이동평균 계산
- 변화율 계산 (MoM, QoQ, YoY)
- 리세션 지표 생성

### 2. 경기 사이클 분류기

4가지 경기 체제로 분류:
1. **Expansion (확장기)**: GDP↑, 실업률↓, 금리↑
2. **Peak (정점)**: GDP 고점, 인플레이션↑, 금리↑
3. **Contraction (수축기)**: GDP↓, 실업률↑, 금리↓
4. **Trough (저점)**: GDP 저점, 실업률 고점

**분류 방법**:
- Rule-based 분류 (전통적 NBER 기준)
- Hidden Markov Model (HMM)
- K-means/GMM 클러스터링

### 3. 체제 기반 시그널

체제에 따른 자산 배분 시그널 생성:

| 체제 | 주식 | 채권 | 원자재 | 현금 |
|------|------|------|--------|------|
| Expansion | 높음 | 낮음 | 중간 | 낮음 |
| Peak | 중간 | 중간 | 높음 | 중간 |
| Contraction | 낮음 | 높음 | 낮음 | 높음 |
| Trough | 중간 | 중간 | 낮음 | 중간 |

## 기술 명세

### 클래스 구조

```python
class MacroIndicatorProcessor:
    """거시경제 지표 처리기"""
    def normalize(self, series): ...
    def calculate_momentum(self, series): ...
    def calculate_diffusion_index(self): ...

class RegimeClassifier(ABC):
    """체제 분류기 기본 클래스"""
    def fit(self, indicators): ...
    def predict(self, indicators): ...
    def get_current_regime(self): ...

class RuleBasedClassifier(RegimeClassifier):
    """규칙 기반 분류기"""

class HMMClassifier(RegimeClassifier):
    """Hidden Markov Model 분류기"""

class RegimeSignalGenerator:
    """체제 기반 시그널 생성기"""
    def generate_allocation_signal(self, regime): ...
    def generate_risk_signal(self, regime): ...
```

### 입력
- FRED 거시경제 지표 시계열
- 분류 파라미터 (임계값, 윈도우 크기 등)

### 출력
- 현재 체제 분류 결과
- 체제 전환 확률
- 자산 배분 시그널
- 리스크 조정 시그널

## 의존성

- pandas, numpy: 데이터 처리
- hmmlearn: Hidden Markov Model
- scikit-learn: 클러스터링
- FRED API 클라이언트 (Phase 1에서 구현)

## 참고자료

- NBER Business Cycle Dating
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series"
- 경기선행지수 (CLI), 경기동행지수 (CCI)
