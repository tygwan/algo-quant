# Quant Finance Domain Research

> algo-quant 프로젝트 발전을 위한 금융 도메인 지식 조사 결과
> 조사일: 2026-02-02 | 도구: Codex CLI (gpt-5, high effort)

---

## 목차

1. [금융 이론 심화](#1-금융-이론-심화)
2. [알고리즘 트레이딩 기술 스택](#2-알고리즘-트레이딩-기술-스택)
3. [리스크 관리 고급 기법](#3-리스크-관리-고급-기법)
4. [머신러닝/딥러닝 in Finance](#4-머신러닝딥러닝-in-finance)
5. [실전 퀀트 전략 유형](#5-실전-퀀트-전략-유형)
6. [규제 및 컴플라이언스](#6-규제-및-컴플라이언스)
7. [프로젝트 로드맵](#7-algo-quant-프로젝트-로드맵)
8. [추천 학습 경로](#8-추천-학습-경로)

---

## 1. 금융 이론 심화

### 1.1 FF5 이후 최신 팩터 모델

#### q-factor model (Hou-Xue-Zhang, 2015) & q5
- **핵심**: 생산기반 자산가격(q-theory)에 근거한 4요인(시장/규모/투자 I/A/수익성 ROE). q5는 기대성장(Expected Growth) 요인 추가
- **수학적 직관**: 기업 최적투자 조건에서 I/A↑ + 기대수익성↑ → 요구수익률↑
- **FF5 대비**: 투자/수익성 요인의 경제적 기반이 강하고 q5는 FF6보다 단면 설명력 높음. 단, 정의 차이(ROE vs RMW)로 구현 민감
- **Python**: `linearmodels`(Fama-MacBeth), `statsmodels`(HAC), pandas/numpy
- **논문**: Hou-Xue-Zhang(2015), Hou-Mo-Xue-Zhang(2021, q5), Hou-Xue-Zhang(2020, Replicating Anomalies)

#### Mispricing Factor Model (Stambaugh & Yuan, 2017)
- **핵심**: MKT + SIZE + 합성요인 2개(MGMT, PERF). 11개 이상현상 랭킹을 공분산 기반 2개 클러스터로 통합
- **수학적 직관**: 다수 이상현상 신호를 저차원 합성요인으로 축약하여 공통 행동적 오정가격 포착
- **FF5 대비**: 이상현상 알파 흡수력 우수, 숏 레그 거래비용/공매도 제한이 구현 리스크
- **논문**: Stambaugh & Yuan(2017, RFS), "Pricing without Mispricing"(2021)

#### DHS Behavioral Factors (Daniel-Hirshleifer-Sun, 2020)
- **핵심**: 시장 + 단기/장기 행동요인. 장기: 발행/자사주 매입 기반, 단기: 이익서프라이즈(PEAD) 기반
- **FF5 대비**: 행동재무의 명시적 구조화, PEAD/발행효과 직접 요인화
- **논문**: DHS(2020, RFS)

### 1.2 자산 가격 결정 이론

| 이론 | 핵심 | 수학적 직관 | Python 도구 |
|------|------|------------|-------------|
| **APT** (Ross 1976) | 무차익거래 + 다요인 선형조합 E[R]=α+β'λ | 선형요인 이탈 → 무위험 차익 → 가격 복원 | `linearmodels`, PCA |
| **ICAPM** (Merton 1973) | 다기간 효용극대화, 헤징수요 반영 추가 요인 | 투자기회 변동 헤지 수요 → 시장 외 λ | FRED 상태변수, 다요인 회귀 |
| **Consumption CAPM** (Breeden 1979) | 소비성장 공분산 → 기대수익, 습관형성으로 퍼즐 개선 | 소비 낮을 때 손실 큰 자산 → 높은 보상 | GMM(linearmodels) |
| **SDF Framework** | E[mR]=1, 요인모형은 m의 선형근사 | m이 카운터사이클일수록 위험보상 | HJ 바운드 검정, Cochrane 교재 |

### 1.3 포트폴리오 이론 심화

| 기법 | 핵심 | Python 라이브러리 |
|------|------|------------------|
| **Black-Litterman** | 시장균형수익 + 투자자 뷰 결합(베이즈) | `PyPortfolioOpt`, `Riskfolio-Lib` |
| **Robust Optimization** | 불확실성 집합 모델링, 최악 경우 보장 | `cvxpy`, `Riskfolio-Lib` |
| **HRP** (Lopez de Prado 2016) | 계층클러스터링 → 재귀적 분할, 역행렬 불필요 | `PyPortfolioOpt`, `Riskfolio-Lib` |
| **NCO** | 클러스터 내 최적화 → 클러스터 간 최적화 | `Riskfolio-Lib`, `skfolio` |
| **Entropy Pooling** (Meucci 2008) | 최소 엔트로피 왜곡으로 유연한 뷰 반영 | `Riskfolio-Lib` |

---

## 2. 알고리즘 트레이딩 기술 스택

### 2.1 퀀트 펀드 인프라
- **리서치/프로덕션 분리**: 백테스트 파이프라인과 실거래 파이프라인 엄격 분리
- **데이터 레이크 + 틱 스토어**: 범용 원천 → 오브젝트 스토리지, 틱/LOB → kdb+/q
- **업계 공통**: C++/Java/Python, kdb+/q, Hadoop/Spark, Kubernetes, Kafka
- **Python**: `polars`, `pyarrow`, `confluent-kafka`, `faust`, `bytewax`, `prefect`

### 2.2 Low-Latency 시스템
- **FPGA**: 피드핸들링/북빌드를 RTL/HLS로 구현, 결정적 지연 확보
- **Kernel Bypass**: DPDK, Solarflare/OpenOnload, AF_XDP로 커널 스택 우회
- **Co-location/DMA**: 거래소 코로 상면, 크로스커넥트, 브로커 DMA
- **역할 분담**: C++/Rust(지연 민감 경로) + Python(리서치/오케스트레이션/모니터링)
- **바인딩**: `pybind11`, `cffi`, `pyo3/maturin`, `numba`, `cython`

### 2.3 Market Microstructure
- **LOB 모델링**: Zero-Intelligence, Queue-Reactive, Hawkes Process
- **스프레드 결정**: 정보 비대칭(Adverse Selection), 재고위험, 처리비용
- **시장충격**: Almgren-Chriss(일시적/영속적), Kyle's Lambda, Propagator
- **최적 실행**: TWAP, VWAP, Implementation Shortfall, POV
- **참고서**: Bouchaud "Trades, Quotes and Prices", O'Hara "Market Microstructure Theory"

### 2.4 데이터 파이프라인

| 기술 | 역할 | 특징 |
|------|------|------|
| **Kafka** | 스트리밍 백본 | 로그 중심, 리플레이, 스케일아웃 |
| **Apache Arrow** | 메모리 포맷 | 컬럼 지향, 제로카피 상호운용 |
| **kdb+/q** | 틱 스토어 | 시계열 특화, 초저지연 쿼리 |

---

## 3. 리스크 관리 고급 기법

### 3.1 Tail Risk

| 기법 | 핵심 | Python |
|------|------|--------|
| **CVaR/ES** | VaR 초과 구간 기대손실, coherent risk measure | `Riskfolio-Lib`, `PyPortfolioOpt` |
| **EVT (GEV, GPD)** | 블록 최대치/임계초과로 꼬리 적합 | `scipy.stats`, `pyextremes` |
| **Tail Hedging** | OTM 풋, VIX 콜로 컨벡서티 확보 | `arch`, `quantlib`, `py_vollib` |

### 3.2 Stress Testing
- **Historical**: 2008 금융위기, COVID-19 등 리스크 팩터 맵핑
- **Hypothetical**: 금리 이동, 주가 갭다운+볼 스파이크 시나리오 그리드
- **Reverse**: 허용 손실 위반 최소 충격 시나리오 탐색 (`cvxpy`, `scipy.optimize`)

### 3.3 Factor Risk Models
- **Barra**: 노출(Exposure) + 팩터수익 + 특이위험 → 공분산 분해. 스타일(밸류/모멘텀/퀄리티/사이즈/볼) + 산업(GICS)
- **자체 구축**: 팩터 정의 → 노출 측정 → 공분산 추정(Ledoit-Wolf, EWMA) → 검증(IC, Turnover)
- **Python**: `sklearn`(PCA, LedoitWolf), `statsmodels`(WLS), `linearmodels`

### 3.4 Position Sizing
- **Kelly Criterion**: 프랙셔널 켈리(0.25-0.5) 권장, 풀 켈리는 과공격적
- **Risk Budgeting**: ERC 등 마진 리스크 기여도 균형화
- **Drawdown Control**: MDD 기반 레버리지 자동 축소, Vol Targeting 병행

---

## 4. 머신러닝/딥러닝 in Finance

### 4.1 시계열 예측 모델

| 모델 | 특징 | 주의점 | Python |
|------|------|--------|--------|
| **LSTM/GRU** | 장단기 의존성 학습 | 비정상성, 레짐변화 취약 | PyTorch, darts |
| **TFT** | 변수선택, 멀티호라이즌, 해석성 | 코변량 분리 설계 필요 | pytorch-forecasting |
| **N-BEATS** | 추세/계절성 분해, interpretable | 단변량 기반 | darts |
| **TimeGPT/Chronos** | Foundation model, 0-shot | 프라이버시, 지연 | nixtla, HuggingFace |
| **XGBoost/LightGBM** | 탭울러 데이터 강건 | 시계열 CV 누수 주의 | xgboost, lightgbm, shap |

### 4.2 강화학습 트레이딩

| 알고리즘 | 특징 | 라이브러리 |
|----------|------|-----------|
| **PPO** | 클리핑 기반 안정적 정책 경사 | Stable-Baselines3, FinRL |
| **SAC** | 최대 엔트로피 RL, 연속 액션 적합 | SB3, RLlib |
| **DDPG** | 오프폴리시 연속 제어 | SB3, FinRL |

- **환경**: FinRL, gymnasium (거래비용/슬리피지/체결모델 내재화)
- **보상 설계**: PnL - λ*TC - DD 패널티

### 4.3 대안 데이터 (Alternative Data)
- **NLP 센티먼트**: FinBERT, GPT 기반 헤드라인/기사 감정 추출 → 엔터티 매핑 → 윈도우 집계
- **위성 이미지**: 주차장 혼잡도, 야간조도 → CNN/YOLO → 자산 매핑
- **소셜 미디어**: Reddit/Twitter 정서 급증, 봇 필터 필수
- **공급망/신용카드**: 매출 nowcast, 프라이버시/MNPI 준수

### 4.4 과적합 방지 (Critical)

| 기법 | 핵심 | Python |
|------|------|--------|
| **Purged K-Fold CV** | 정보누수 제거, 엠바고 | `mlfinlab` |
| **CPCV** | 조합적 홀드아웃, 성능 분포 추정 | `mlfinlab` |
| **Walk-Forward** | 롤링 학습/검증, 레짐 변화 대응 | `vectorbt`, `backtrader` |
| **Deflated Sharpe Ratio** | 다중시도 보정 후 유의성 | `mlfinlab` |
| **Multiple Testing** | White's Reality Check, FDR(BH) | `arch`, `statsmodels` |

---

## 5. 실전 퀀트 전략 유형

### 5.1 Statistical Arbitrage
- **Pairs Trading**: Cointegration(Engle-Granger, Johansen) → 스프레드 → 밴드 진입/청산
- **Mean Reversion**: OU 프로세스 dX=θ(μ-X)dt+σdW, 하프라이프 기반 신호
- **Distance/Copula**: 공적분 대안, 코퓰라로 의존구조 모델링

### 5.2 Momentum 전략
- **Cross-Sectional**: 3-12M 수익률 랭킹 롱숏 (Jegadeesh & Titman 1993)
- **Time-Series**: 각 자산 자체 추세, 볼 스케일링 (Moskowitz et al. 2012)
- **Dual Momentum**: 절대 + 상대 모멘텀 결합 (Antonacci)
- **Crash Hedging**: 리버설 급락 리스크 → 풋/타겟볼 컨트롤

### 5.3 Market Making
- **Avellaneda-Stoikov**: 인벤토리/주문흐름 기반 최적 호가
- **Inventory Risk**: 재고 한도/감마 조절, 볼-스케일드 스프레드
- **Spread Optimization**: 로지스틱/포아송 체결 모델, 동적 스프레드

### 5.4 Event-Driven
- **PEAD**: 실적 서프라이즈 후 드리프트 (Ball & Brown 1968)
- **M&A Arbitrage**: 딜 스프레드 캡처, 실패 리스크 관리
- **Macro Events**: 금리/고용 서프라이즈 이벤트 윈도우 트레이딩

### 5.5 Volatility 전략
- **Vol Arbitrage**: Implied vs Realized 프리미엄 수취, 델타/베가 헤지
- **Variance Swap**: 정방분산 거래, 옵션 스트립 복제
- **VIX 전략**: 선물 커브 콘탱고/백워데이션 캐리
- **Dispersion**: 지수 옵션 vs 구성종목 옵션 상관 분산 차이

---

## 6. 규제 및 컴플라이언스

### 6.1 미국 규제

| 규제 | 핵심 요건 |
|------|----------|
| **SEC Rule 15c3-5** | Pre-trade risk controls, CEO 연 1회 인증, naked sponsored access 금지 |
| **Reg NMS** | Rule 611(트레이드스루 금지), Rule 605/606(체결/라우팅 공시) |
| **Reg SHO** | 공매도 Locate/Close-out, Alternative Uptick(10% 하락 시) |
| **Dodd-Frank** | FSOC 시스템리스크 감독, 파생상품 중앙청산/보고(Title VII) |
| **SEC 2024-25 동향** | AI washing 단속, 15c3-5 점검 강화, LULD 고도화 |

### 6.2 유럽 규제

| 규제 | 핵심 요건 |
|------|----------|
| **MiFID II/RTS 6** | 알고리즘 사전 승인, 프리트레이드 통제, Kill Switch 의무 |
| **RTS 9 (OTR)** | 주문대체결비율 감시, 상한 초과 시 조치 |
| **HFT 규정** | 인가/감독 대상, 5년 이상 기록 보관 |

### 6.3 한국 규제
- **가상자산 이용자 보호법** (2024.7.19 시행): 자산 분리, 불공정거래 규제
- **KRX 시장감시**: 허수성 주문/레이어링 금지, 종가집중매매 감시
- **ATS(넥스트레이드)**: 2025 등장, 시장점유/거래량 한도 규제 논의 중
- **개인 vs 기관**: DMA는 인가 주체 중심, 개인은 브로커 API 간접접근

### 6.4 공통 의무사항 체크리스트
- Pre-trade risk controls: 가격콜라, 수량한도, SMP, 스로틀링
- Circuit breakers: LULD(미), 회로차단(유), 서킷브레이커(한)
- Recordkeeping: SEC 17a-4(WORM), MiFIR Art.25(5년)
- 시장조작 방지: Spoofing, Layering 금지

---

## 7. algo-quant 프로젝트 로드맵

### 단기 (1-3개월)

| 기능 | 난이도 | 효과 | 위치 |
|------|--------|------|------|
| Pre-trade Risk 모듈 | 중 | 높음 | `risk_manager.py`, `executor.py` |
| Kill Switch | 낮음 | 높음 | `executor.py`, `broker.py` + Dash UI |
| OTR/Cancel Ratio 모니터 | 중 | 중 | `realtime.py`, UI |
| 감사로그/Audit Trail | 중 | 높음 | `logs/` append-only |
| LULD/서킷브레이커 시뮬 | 낮음 | 중 | `backtest/engine.py` |
| Best Execution 리포트 | 중 | 중 | `backtest/metrics.py` |

### 중기 (3-6개월)

| 기능 | 난이도 | 효과 | 기술 |
|------|--------|------|------|
| Smart Order Router (SOR) | 높음 | 높음 | 마이크로구조, CCXT |
| 실시간 이상거래 탐지 | 높음 | 높음 | PyFlink/Kafka, ML |
| ML 파이프라인 통합 | 중 | 높음 | MLflow, Feast, DVC |
| 데이터 거버넌스/품질 | 중 | 중 | Great Expectations, DuckDB |

### 장기 (6-12개월)

| 기능 | 난이도 | 효과 | 목표 |
|------|--------|------|------|
| 이벤트 중심 아키텍처 | 높음 | 높음 | 마이크로서비스, Kafka 버스 |
| 멀티에셋 확장 | 높음 | 높음 | 옵션/선물/채권 그릭스 |
| 규제지역별 정책엔진 | 중 | 높음 | US/EU/KR 규제 프로파일 |
| 운영/감사 준비 | 중 | 높음 | 변경관리, DR, 접근통제 |

---

## 8. 추천 학습 경로

### 입문

| 카테고리 | 리소스 |
|----------|--------|
| **책** | Python for Finance (Hilpisch), Quantitative Trading (Ernest Chan) |
| **강좌** | Columbia "Financial Engineering and Risk Management", Georgia Tech "ML for Trading" |
| **논문** | Fama & French(1993), Carhart(1997) |

### 중급

| 카테고리 | 리소스 |
|----------|--------|
| **책** | Advances in Financial Machine Learning (Lopez de Prado), Active Portfolio Management (Grinold & Kahn) |
| **강좌** | Andrew Ng Deep Learning, 시계열/베이지안 MOOC |
| **논문** | Probability of Backtest Overfitting (Lopez de Prado & Bailey) |

### 고급

| 카테고리 | 리소스 |
|----------|--------|
| **책** | Market Microstructure Theory (O'Hara), Designing Data-Intensive Applications (Kleppmann), Asset Pricing (Cochrane) |
| **주제** | 마이크로구조, 저지연 시스템, 규제준수(17a-4/RTS 6/RTS 25) |
| **실습** | Qlib/Lean/MLflow/Feast 리서치→프로덕션 파이프라인 |

### 핵심 Python 라이브러리 종합

| 영역 | 라이브러리 |
|------|-----------|
| **팩터/회귀** | `linearmodels`, `statsmodels`, `pandas`, `numpy` |
| **포트폴리오** | `PyPortfolioOpt`, `Riskfolio-Lib`, `cvxpy`, `skfolio` |
| **백테스트** | `vectorbt`, `backtrader`, `zipline-reloaded` |
| **ML/DL** | `PyTorch`, `LightGBM`, `XGBoost`, `darts`, `mlfinlab` |
| **강화학습** | `Stable-Baselines3`, `FinRL`, `gymnasium` |
| **NLP** | `transformers`(FinBERT), `spacy`, `nltk` |
| **리스크** | `arch`, `pyextremes`, `quantlib`, `py_vollib` |
| **데이터** | `yfinance`, `pandas-datareader`, `polars`, `pyarrow` |
| **스트리밍** | `confluent-kafka`, `faust`, `bytewax` |
| **품질/실험** | `great-expectations`, `MLflow`, `Feast`, `shap` |

---

## 핵심 논문/책 참고 목록

### 팩터 모델
- Fama & French (1993, 2015) - FF3, FF5
- Hou, Xue, Zhang (2015) - q-factor model
- Hou, Mo, Xue, Zhang (2021) - q5
- Stambaugh & Yuan (2017) - Mispricing factors
- Daniel, Hirshleifer, Sun (2020) - DHS behavioral factors

### 자산가격 이론
- Ross (1976) - APT
- Merton (1973) - ICAPM
- Breeden (1979) - Consumption CAPM
- Campbell & Cochrane (1999) - Habit formation
- Hansen & Jagannathan (1991) - SDF bounds
- Cochrane (2005) - Asset Pricing (교재)

### 포트폴리오
- Black & Litterman (1992) - BL model
- Lopez de Prado (2016) - HRP
- Lopez de Prado (2019) - NCO
- Meucci (2008) - Entropy Pooling

### 미시구조/실행
- Almgren & Chriss (2001) - Optimal execution
- Kyle (1985) - Lambda
- Avellaneda & Stoikov (2008) - Market making
- Bouchaud - "Trades, Quotes and Prices"

### ML/과적합
- Lopez de Prado (2018) - "Advances in Financial Machine Learning"
- Bailey & Lopez de Prado (2014) - Deflated Sharpe Ratio
- Grinold & Kahn - "Active Portfolio Management"

### 리스크
- McNeil, Frey, Embrechts - "Quantitative Risk Management"
- Roncalli - "Introduction to Risk Parity"

---

*본 문서는 법률 자문이 아니며, 규제 해석/적용은 관할 규제기관/법률전문가 확인이 필요합니다.*
