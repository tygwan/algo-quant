# algo-quant Project Memory

## Critical: Plotly 6.x + Dash 3.x Binary Encoding Issue
- **Plotly 6.5.1** serializes numpy arrays as `{dtype: 'f8', bdata: '...'}` (binary encoding)
- **Dash 3.3.0** may not decode this correctly, causing **empty charts**
- **Fix**: Always convert to Python lists before passing to `go.Scatter(y=series.tolist())` and `go.Scatter(x=list(index))`
- Helper: `_to_list()` function in `live_analyzer.py`

## yfinance Interval Constraints
| Interval | Max Period | Valid yfinance periods |
|----------|-----------|----------------------|
| 1m | 7 days | 1d, 5d |
| 5m | 60 days | 1d, 5d, 1mo |
| 15m | 60 days | 1d, 5d, 1mo |
| 60m | ~6 months | 1d, 5d, 1mo, 3mo, 6mo |
| 1d+ | unlimited | 1mo, 3mo, 6mo, 1y, 2y, 5y |

## yfinance Bulk Download Column Format
- `yf.download()` returns **MultiIndex** columns: Level 0 = PriceType (Close, Open...), Level 1 = Ticker
- Single ticker still uses MultiIndex in newer versions
- Need to handle both flat and MultiIndex column access

## Performance: Data Fetching
- Sequential yfinance fetching: ~1-3s per ticker
- `yf.download()` bulk: single HTTP request, much faster
- Fallback: `ThreadPoolExecutor(max_workers=8)` for parallel individual fetch
- 8 tickers bulk: ~1.5-2s vs ~16-24s sequential

## Dashboard Stack
- Dash 3.3.0 + Plotly 6.5.1 + Python 3.11+
- `suppress_callback_exceptions=True` - errors silently swallowed
- Package manager: `uv` (not `pip`)
- Run: `uv run python scripts/run_dashboard.py`
- Tests: `uv run pytest tests/ -v`

## CSS: Dropdown Overflow Fix
- `.chart-container` with `backdrop-filter` creates stacking context
- Dropdowns in upper containers get hidden behind lower containers
- Fix: add `position: relative; z-index: 10; overflow: visible` to input section
