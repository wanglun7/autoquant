# M001 60d 横截面动量日调仓

## 基本信息

- 实验编号：`M001`
- 实验名称：`60d cross-sectional momentum, Top 50, daily rebalance`
- 日期：`2026-03-13`
- 负责人：Codex

## 假设

- 这次想验证的市场假设是什么？
  - 如果 Binance 永续上的横截面中期趋势有效，那么把 `120d` 缩短到 `60d`，可能会在保持一定趋势强度的同时，提高对市场变化的响应速度。
- 这次只改变了哪一个变量？
  - 只改变了动量窗口：`120d -> 60d`

## 策略定义

- 市场：Binance `USDT-M perpetual`
- 数据频率：`4h`
- 信号：过去 `60d` 收益，等于 `360` 根 `4h` bar
- 调仓频率：`日调仓`
- 样本池：过去 `60d` 平均 `quote_volume` 的 `Top 50`
- 持仓规则：横截面做多前 `20%`，做空后 `20%`，双边等权，`gross = 1.0`
- 成本口径：`fee + slippage + funding`

## 配置与结果文件

- 配置文件：[m001_cs_mom_60d_daily.yaml](/Users/lun/Desktop/manifex/quant/configs/m001_cs_mom_60d_daily.yaml)
- 结果目录：[m001_cs_mom_60d_daily](/Users/lun/Desktop/manifex/quant/results/m001_cs_mom_60d_daily)
- summary 文件：[summary.json](/Users/lun/Desktop/manifex/quant/results/m001_cs_mom_60d_daily/summary.json)

## 核心结果

| 指标 | 数值 |
|---|---:|
| 年化收益 | `-4.63%` |
| Sharpe | `-0.008` |
| 最大回撤 | `-58.29%` |
| 毛收益累计 | `1.44%` |
| 净收益累计 | `-21.12%` |
| 平均换手 | `0.0382` |

## 对比结论

- 当前冠军版本：`B001 / sample_momentum`
- 是否跑赢当前冠军：否
- 是否仍为净收益为正：否
- 成本放大后是否还能活：否，`2x` 成本后净收益进一步恶化

## 判断

- 实验状态：`rejected`
- 结论一句话：`60d` 在当前 Binance 永续全量研究口径下太弱，既没有足够的毛收益，也没有成本防御力。
- 失败原因或成功原因：
  - 相比 `M002 / 120d`，更短窗口没有换来更高毛收益
  - 成本和 funding 侵蚀后，净值明显转负
  - 2022 年拖累很大，整体稳健性不足

## 下一步动作

- 下一条要跑什么：`M003 / 180d 横截面动量，Top 50，日调仓`
- 为什么是它：当前窗口层只剩 `180d` 还没验证，需要看更长趋势是否比 `120d` 更稳、更能覆盖成本
