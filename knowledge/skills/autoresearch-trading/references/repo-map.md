# Repo Map

## Mutable During Normal Research

- `src/binance4h_research/trading_autoresearch/strategy.py`

## Fixed During Normal Research

- `src/binance4h_research/trading_autoresearch/prepare_market.py`
- `src/binance4h_research/trading_autoresearch/evaluate.py`
- `src/binance4h_research/trading_autoresearch/runner.py`
- `configs/trading_autoresearch.yaml`
- `docs/program_trading.md`

## Main Commands

```bash
PYTHONPATH=src python3 -m binance4h_research build-trading-context --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research show-trading-champions --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research replay-trading-run --program configs/trading_autoresearch.yaml --run-id <run_id>
```

## Result Paths

- `results/trading_autoresearch/trading_autoresearch_v1/results.tsv`
- `results/trading_autoresearch/trading_autoresearch_v1/champions.json`
- `results/trading_autoresearch/trading_autoresearch_v1/runs/`
