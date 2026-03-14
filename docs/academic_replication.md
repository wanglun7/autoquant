# Academic Replication

This subsystem reproduces three weekly cross-sectional crypto momentum constructions:

- `ltw`: Liu, Tsyvinski, Wu CMOM factor
- `grobys`: plain equal-weighted top-30 momentum
- `ficura`: size/liquidity split momentum quintiles

## Design boundaries

- This is a spot cross-section replication layer, not a Binance perpetual strategy layer.
- Data is fetched from CoinGecko and cached locally.
- Weekly portfolios use a `paper_52w` calendar: 52 weeks per year, first 51 weeks have 7 days, the final week contains the remainder.
- Stablecoins are excluded via explicit config lists.

## Important approximations

- CoinGecko public data may not include the same inactive universe used by original CoinMarketCap studies.
- LTW and Fičura operate on the weekly panel directly.
- Grobys uses daily closes for the 30-day formation window and 1-day skip, then holds for the following paper week.
- Fičura uses equal-weighted Q5-Q1 returns and caps next-week simple returns at `1000%`.

## CLI

```bash
binance4h fetch-academic-data --coin-ids bitcoin ethereum ripple --start-date 2014-01-01 --end-date 2018-12-31
binance4h build-academic-panel --config configs/academic_ltw.yaml
binance4h run-paper-replication --config configs/academic_ltw.yaml
binance4h compare-replications --configs configs/academic_ltw.yaml configs/academic_grobys.yaml configs/academic_ficura.yaml
```
