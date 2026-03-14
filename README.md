# Binance 4h Research

Minimal research framework for Binance USDT-M perpetual 4h cross-sectional strategies.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
binance4h run-backtest --config configs/sample_momentum.yaml
```

## CLI

```bash
binance4h fetch-data --data-dir data/raw --symbols BTCUSDT ETHUSDT
binance4h build-universe --config configs/sample_momentum.yaml
binance4h run-backtest --config configs/sample_momentum.yaml
binance4h report --results-dir results/sample_momentum
```

## Binance Paper Approx

```bash
binance4h run-paper-approx --config configs/paper_ltw_binance.yaml
binance4h run-paper-approx --config configs/paper_grobys_binance.yaml
binance4h run-paper-approx --config configs/paper_ficura_binance.yaml
binance4h compare-paper-approx --configs configs/paper_ltw_binance.yaml configs/paper_grobys_binance.yaml configs/paper_ficura_binance.yaml
```

## Academic Replication

```bash
binance4h fetch-academic-data --coin-ids bitcoin ethereum ripple --start-date 2014-01-01 --end-date 2018-12-31
binance4h build-academic-panel --config configs/academic_ltw.yaml
binance4h run-paper-replication --config configs/academic_ltw.yaml
binance4h compare-replications --configs configs/academic_ltw.yaml configs/academic_grobys.yaml configs/academic_ficura.yaml
```

## Trading Autoresearch

This path follows the Karpathy-style loop: keep the evaluator fixed, edit only `src/binance4h_research/trading_autoresearch/strategy.py`, and record each run.

```bash
PYTHONPATH=src python3 -m binance4h_research build-trading-context --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research run-trading-autoresearch-batch --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research show-trading-champions --program configs/trading_autoresearch.yaml
PYTHONPATH=src python3 -m binance4h_research replay-trading-run --program configs/trading_autoresearch.yaml --run-id <run_id>
```

## Strategy assumptions

- Market: Binance USDT-M perpetual futures
- Frequency: 4h bars
- Signal timing: previous completed bar close
- Fill timing: next bar open
- Costs: fees + slippage + funding
