# Phase 6: Production (In Progress)

## 목표
실시간 데이터 파이프라인 및 자동 트레이딩 시스템

> **Status (2026-02-05)**: 실행 엔진/실시간 스트림/리밸런싱/페이퍼 브로커 기반 구현 완료, 실거래 검증 및 알림 시스템 진행 중

## 범위

### 포함
- 실시간 데이터 파이프라인
- 자동 리밸런싱
- 브로커 연동 (Binance, 증권사 API)
- 알림 시스템

### 제외
- 고빈도 트레이딩 (HFT)
- 복잡한 주문 유형

## 기술 상세

### 실시간 파이프라인
```
[Data Source] → [Stream Processor] → [Signal Generator] → [Order Manager]
     │                  │                    │                   │
     └── WebSocket ────┴─── Redis/Kafka ────┴─── Database ──────┘
```

### 자동 리밸런싱
- 일간/주간/월간 리밸런싱 스케줄
- 목표 vs 실제 포지션 차이 모니터링
- 자동 주문 생성

### 브로커 연동
```python
class BrokerInterface(ABC):
    @abstractmethod
    def get_positions(self) -> Dict[str, float]: ...

    @abstractmethod
    def place_order(self, symbol: str, qty: float, side: str) -> Order: ...

    @abstractmethod
    def get_balance(self) -> float: ...
```

## 완료 조건

- [ ] 실시간 데이터 수신 안정화 (24h)
- [x] 자동 리밸런싱 기본 동작
- [ ] 브로커 연동 통합 테스트 완료
- [ ] 페이퍼 트레이딩 검증
- [ ] 알림 시스템 구축

## 예상 산출물

```
src/execution/
├── __init__.py
├── broker.py
├── executor.py
├── realtime.py
├── rebalancer.py
└── notifier.py (planned)
```

## 주의사항

- 실제 자금 투입 전 충분한 페이퍼 트레이딩
- API 키 보안 관리 필수
- 리스크 관리 규칙 엄격 적용
