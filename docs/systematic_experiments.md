# 系统化实验台账

## 目标

第一阶段目标只有一个：

- 在 Binance `USDT-M perpetual` 上，找到`扣成本后净收益为正`的横截面动量版本。

当前不追求：

- 同时优化 funding filter / regime filter
- 学术论文近似复现
- 高频或复杂模型

## 固定研究口径

- 市场：Binance `USDT-M perpetual`
- 原始数据频率：`4h`
- 回测收益口径：`open-to-open`
- 成本口径：`fee + slippage + funding`
- 调仓：按配置控制，优先比较 `4h / 日 / 周`
- 样本池：动态流动性筛选，默认用过去一段时间的 `quote_volume`

## 当前基准

### 冠军基准

当前冠军仍然是 `sample_momentum`，原因不是它足够好，而是目前它是已跑结果里净收益最高的版本。

| 实验编号 | 实验名 | 年化 | Sharpe | 最大回撤 | 毛收益累计 | 净收益累计 | 平均换手 | 结论 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `B001` | `sample_momentum` | `5.26%` | `0.320` | `-50.23%` | `715.75%` | `29.25%` | `0.2802` | 净收益为正，但回撤大、换手高、质量不够 |
| `B002` | `cs_mom_120d_daily` | `-0.58%` | `0.111` | `-52.39%` | `15.73%` | `-2.87%` | `0.0266` | 成本明显下降，但净收益仍为负 |

### 当前判断

- `sample_momentum` 说明：纯横截面动量在 Binance 永续上不是完全没东西。
- `cs_mom_120d_daily` 说明：把换手压下去以后，成本改善明显，但信号强度还不够。
- 第一阶段主线不再扩散，先把中长横截面动量这条线研究透。

## 实验决策规则

### 状态定义

- `planned`：已列入矩阵，未执行
- `running`：正在跑
- `completed`：已完成，等待与冠军比较
- `promoted`：成为新的冠军或进入下一轮主线
- `rejected`：证据不足，不继续

### 升级规则

- `promoted`
  - 净收益高于当前冠军，且 Sharpe 没有明显恶化
- `rejected`
  - 净收益为负，或成本放大后快速崩掉
- `completed`
  - 有一定信息量，但不足以升级为冠军

## 第一阶段实验矩阵

### A. 窗口层

| 编号 | 变量 | 配置目标 | 当前状态 | 结果摘要 |
|---|---|---|---|---|
| `M001` | `60d` 动量 | `Top 50`，日调仓 | `rejected` | 年化 `-4.63%`，净收益累计 `-21.12%`，明显弱于 `M002` |
| `M002` | `120d` 动量 | `Top 50`，日调仓 | `completed` | 当前结果为负，保留作比较基准 |
| `M003` | `180d` 动量 | `Top 50`，日调仓 | `planned` | 待跑 |

### B. 调仓层

这一层只在窗口层选出当前最优窗口后再继续。

| 编号 | 变量 | 配置目标 | 当前状态 | 结果摘要 |
|---|---|---|---|---|
| `M004` | 高频调仓 | 最优窗口 + `4h` 调仓 | `planned` | 待跑 |
| `M005` | 中频调仓 | 最优窗口 + `日调仓` | `planned` | 待跑 |
| `M006` | 低频调仓 | 最优窗口 + `周调仓` | `planned` | 待跑 |

### C. Universe 层

这一层只在调仓层后继续。

| 编号 | 变量 | 配置目标 | 当前状态 | 结果摘要 |
|---|---|---|---|---|
| `M007` | 窄池 | 最优窗口 + `Top 30` | `planned` | 待跑 |
| `M008` | 中池 | 最优窗口 + `Top 50` | `planned` | 待跑 |
| `M009` | 宽池 | 最优窗口 + `Top 80` | `planned` | 待跑 |
| `M010` | 大流动性层 | 最优窗口 + `large-liquid only` | `planned` | 待跑 |

### D. 稳健性层

只对冠军版本做。

| 编号 | 变量 | 配置目标 | 当前状态 | 结果摘要 |
|---|---|---|---|---|
| `M011` | 成本压力 | 冠军版本 + `2x` 成本 | `planned` | 待跑 |
| `M012` | 成本压力 | 冠军版本 + `3x` 成本 | `planned` | 待跑 |
| `M013` | 分年表现 | 冠军版本 + 年度拆分 | `planned` | 待跑 |
| `M014` | 分腿表现 | 冠军版本 + long/short 拆分 | `planned` | 待跑 |

## 配置与结果映射

| 编号 | 配置文件 | 结果目录 | 当前状态 |
|---|---|---|---|
| `B001` | [sample_momentum.yaml](/Users/lun/Desktop/manifex/quant/configs/sample_momentum.yaml) | [sample_momentum](/Users/lun/Desktop/manifex/quant/results/sample_momentum) | `promoted` |
| `B002` | [cs_mom_120d_daily.yaml](/Users/lun/Desktop/manifex/quant/configs/cs_mom_120d_daily.yaml) | [cs_mom_120d_daily](/Users/lun/Desktop/manifex/quant/results/cs_mom_120d_daily) | `completed` |
| `M001` | [m001_cs_mom_60d_daily.yaml](/Users/lun/Desktop/manifex/quant/configs/m001_cs_mom_60d_daily.yaml) | [m001_cs_mom_60d_daily](/Users/lun/Desktop/manifex/quant/results/m001_cs_mom_60d_daily) | `rejected` |

## 当前最佳版本

- 当前冠军：`B001 / sample_momentum`
- 原因：在已记录结果里，仍然是净收益最高的版本
- 主要问题：
  - 换手过高
  - 最大回撤过大
  - 结果质量不足以上线

## 下一步待跑

严格按顺序，不并行扩散：

1. `M003`：`180d` 横截面动量，`Top 50`，日调仓
2. 比较 `M001 / M002 / M003`，选一个窗口进入调仓层
3. 用该窗口跑 `M004 / M005 / M006`
4. 如果三条窗口都不行，再考虑 funding/regime 作为第二阶段变量

## 记录规范

- 每次实验只改一个变量
- 新实验必须补一条单实验记录
- 如果结果没有超过冠军，也必须写明失败原因
- 不允许只记录“好看结果”，淘汰结果同样要记
