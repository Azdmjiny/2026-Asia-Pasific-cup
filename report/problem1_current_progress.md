# Problem 1 当前建模进程报告  
## APMCM A题：出厂水浊度 NTU 预测模型阶段总结

> 当前版本对应 `p1_4.ipynb`。  
> 本报告用于总结第一题目前已经完成的建模流程、关键改进、模型结果、预测结果、变量解释和后续建议。  
> 当前阶段暂未将周期性特征和 drop phase 特征纳入模型，因为从现有图形判断，周期性和突降阶段虽然存在一定迹象，但还不够稳定和明确，暂时不适合作为最终模型特征直接加入。

---

# 1. 当前建模目标

第一题的核心目标是预测出厂水浊度：

```text
target_NTU = 当前时刻出厂水浊度
```

当前模型不再使用所有同期变量，也不使用大量冗余 lag 特征，而是采用：

```text
selected-one-lag 特征集
```

即：每个候选变量只保留一个最重要的滞后阶数，再用于模型训练。

当前模型形式可以概括为：

```text
NTU_t = f(
    FILT. NTU_lag1,
    CLR_lag0,
    R/W FLOW_lag1,
    T/W FLOW_lag1,
    RIVER LEVEL_lag0,
    R/W NTU_lag0,
    CL2_lag0,
    ALUM_lag1,
    F/RIDE_lag6
)
```

其中：

```text
lag0 = 当前时刻
lag1 = 2 小时前
lag6 = 12 小时前
```

---

# 2. 当前 notebook 的主要改进

相比之前的 baseline 和 selected-lag 初版，`p1_4.ipynb` 主要有以下改进：

## 2.1 加入 NTU 极端值 clipping

当前版本加入了：

```python
CLIP_NTU_UPPER = 2.0
CLIP_ALL_NTU_RELATED_COLUMNS = True
```

含义是：

```text
所有 NTU 相关列中，大于 2 的值都会被映射为 2。
```

数学表达为：

```text
NTU_clipped = min(NTU, 2)
```

处理对象包括：

```text
target_NTU
FILT. NTU_lag1
R/W NTU_lag0
其他列名中含 NTU 的变量
```

该步骤属于：

```text
数据预处理 / 极端值处理 / 上限截断
```

不是模型训练参数。

它的作用是：

```text
1. 降低极端浊度尖峰对模型训练的影响；
2. 减少少数异常高值对 RMSE 的拉高；
3. 使模型更关注正常运行区间内的 NTU 变化；
4. 提高模型在常规工况下的稳定性。
```

注意：如果论文最终采用该版本，需要明确说明：

```text
模型预测对象是经过 NTU 上限截断处理后的出厂水浊度。
```

不能写成完全没有处理过的原始 NTU 预测。

---

## 2.2 F/RIDE 缺失值填 0

当前版本对 F/RIDE 相关 lag 特征进行缺失值处理：

```text
F/RIDE 缺失值 → 0
```

理由是：

```text
1. F/RIDE 缺失率较高；
2. 如果直接 dropna，会导致大量样本被删除；
3. F/RIDE 是辅助化学指标，不适合作为强核心变量；
4. 填 0 后可以保留样本规模，同时减少该变量对模型训练流程的破坏。
```

当前最终特征中保留的是：

```text
F/RIDE_lag6
```

即 12 小时前的 F/RIDE。

但从后续 feature importance 结果看，F/RIDE 的贡献较低。因此论文中应将其表述为：

```text
辅助水质指标
```

而不是核心控制变量。

---

## 2.3 使用 RIVER LEVEL 替代 C/W WELL LEVEL

当前版本已经将：

```text
C/W WELL LEVEL
```

替换为：

```text
RIVER LEVEL
```

其含义是从厂内清水池水位变量转向原水侧环境变量。

这样做的理由是：

```text
1. RIVER LEVEL 更能反映原水侧水源环境状态；
2. 河流水位变化可能影响原水浊度和进水状态；
3. 对于出厂水浊度预测，RIVER LEVEL 可以作为外部环境背景变量。
```

当前最终选择结果是：

```text
RIVER LEVEL_lag0
```

即当前时刻河流水位。

---

## 2.4 新增 GAM 对比模型

当前版本在原有模型基础上新增了：

```text
Selected-one-lag GAM
```

当前模型对比包括：

```text
1. Selected-one-lag Random Forest
2. Selected-one-lag XGBoost
3. Selected-one-lag GAM
```

其中 GAM 使用的是：

```text
SplineTransformer + Ridge
```

可以理解为一种 spline-based generalized additive model。

模型形式为：

```text
NTU_t = β0 + f1(x1) + f2(x2) + ... + fp(xp) + ε
```

GAM 的加入价值是：

```text
1. 不再只有树模型对比；
2. 增加了一个可解释性更强的非线性基准模型；
3. GAM 能刻画单变量非线性关系；
4. GAM 结构比 RF/XGBoost 更受约束；
5. 如果 RF 显著优于 GAM，则说明变量间可能存在交互效应或非加性结构。
```

---

## 2.5 暂未加入周期性和 drop phase 特征

当前虽然在 NTU 时间序列图中观察到一定周期迹象，并且存在类似：

```text
一段时间后快速下降
```

的局部现象，但当前版本暂未将其显式加入模型。

暂不加入的原因是：

```text
1. 周期性信号并不够稳定；
2. 快速下降阶段的时间位置还没有被严格量化；
3. 如果人为构造 drop flag，可能带来主观性；
4. 当前模型加入 clipping 和 GAM 后性能已经明显提升；
5. 在没有更强证据前，暂时不引入 OP_STEP / DAY_SIN / DAY_COS / DROP_PHASE_FLAG。
```

因此，当前模型仍然保持：

```text
selected-one-lag 变量驱动模型
```

而不是：

```text
selected-one-lag + explicit periodic features
```

这使模型解释更简洁，避免过度特征工程。

---

# 3. 当前数据处理流程

当前 `p1_4.ipynb` 的完整流程如下：

```text
读取 selected_one_lag_model_data.xlsx
↓
将 NTU 相关列中大于 2 的值映射为 2
↓
将 F/RIDE 相关缺失值填为 0
↓
排除三个目标预测 OP_DATE
↓
按时间顺序划分训练集与测试集
↓
使用训练集统计量进行 Z-score 标准化
↓
使用训练集中位数进行缺失值插补
↓
训练 RF / GAM / XGBoost
↓
评估 MAE / RMSE / R² / MAPE
↓
对 2026-02-01、2026-02-10、2026-02-20 进行预测
↓
输出模型结果、预测结果、特征重要性和可视化图
```

---

# 4. 当前最终特征集

当前模型使用 9 个 selected-one-lag 特征：

| 变量 | 最终滞后项 | 时间含义 |
|---|---:|---|
| `FILT. NTU` | `lag1` | 2 小时前滤后水浊度 |
| `CLR` | `lag0` | 当前余氯 / 相关水质指标 |
| `R/W FLOW` | `lag1` | 2 小时前原水流量 |
| `T/W FLOW` | `lag1` | 2 小时前处理水流量 |
| `RIVER LEVEL` | `lag0` | 当前河流水位 |
| `R/W NTU` | `lag0` | 当前原水浊度 |
| `CL2` | `lag0` | 当前加氯 / 余氯相关指标 |
| `ALUM` | `lag1` | 2 小时前矾投加量 |
| `F/RIDE` | `lag6` | 12 小时前 F/RIDE |

对应标准化后进入模型的列为：

```text
FILT. NTU_lag1_z
CLR_lag0_z
R/W FLOW_lag1_z
T/W FLOW_lag1_z
RIVER LEVEL_lag0_z
R/W NTU_lag0_z
CL2_lag0_z
ALUM_lag1_z
F/RIDE_lag6_z
```

---

# 5. 训练集与测试集划分

当前采用时间顺序划分，而不是随机划分：

```text
训练集：前 80%
测试集：后 20%
```

当前样本量为：

| 数据部分 | 样本数 |
|---|---:|
| 训练集 | 4099 |
| 测试集 | 1025 |
| 特征数 | 9 |

采用时间顺序划分的原因是：

```text
1. 水质数据具有时间序列属性；
2. 随机划分会使未来信息和过去信息混合；
3. 随机划分可能导致测试结果虚高；
4. 时间顺序划分更接近真实预测场景。
```

---

# 6. 当前模型结果

## 6.1 模型评价指标

当前 `p1_4.ipynb` 的模型评价结果如下：

| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Selected-one-lag Random Forest | **0.162865** | **0.216208** | **0.392095** | 48.743367% |
| Selected-one-lag XGBoost | 0.160152 | 0.224153 | 0.346598 | **45.401629%** |
| Selected-one-lag GAM | 0.181429 | 0.224506 | 0.344542 | 51.885268% |

---

## 6.2 结果解释

从 MAE 看：

```text
XGBoost 的 MAE 最低，为 0.160152。
```

但从 RMSE 和 R² 看：

```text
Random Forest 表现最好。
```

其中 Random Forest：

```text
RMSE = 0.216208
R²   = 0.392095
```

说明经过 selected-one-lag 特征、NTU clipping、F/RIDE 缺失处理和标准化后，Random Forest 已经具有较明显预测能力。

相比之前版本中 RF 的 R² 只有约 0.0756，当前增强版的 R² 提升到约 0.3921，说明：

```text
NTU clipping + 变量调整 + GAM 对比框架后的建模版本明显更稳定。
```

但需要注意：

```text
R² = 0.3921 仍不是非常高。
```

因此论文里不能写：

```text
模型预测非常准确。
```

更合适的表述是：

```text
模型在测试集上取得了中等程度的解释能力，并且相比 baseline 和初始 selected-lag 模型有明显改进。
```

---

## 6.3 最终主模型选择

当前建议选择：

```text
Selected-one-lag Random Forest
```

作为第一题主模型。

理由是：

```text
1. RMSE 最低；
2. R² 最高；
3. 对异常值和非线性关系较稳健；
4. 特征重要性结果清晰；
5. 比 GAM 和 XGBoost 更适合作为最终解释模型。
```

虽然 XGBoost 的 MAE 更低，但：

```text
XGBoost 的 RMSE 高于 RF；
XGBoost 的 R² 低于 RF。
```

说明 XGBoost 在整体误差稳定性上略弱于 Random Forest。

因此最终可写：

```text
Random Forest was selected as the final predictive model because it achieved the lowest RMSE and the highest R² among the compared models.
```

---

# 7. GAM 对比模型的意义

当前 GAM 结果为：

```text
MAE  = 0.181429
RMSE = 0.224506
R²   = 0.344542
```

GAM 的表现略弱于 RF，但和 XGBoost 接近。

这说明：

```text
1. 变量与 NTU 之间确实存在非线性关系；
2. 单变量加性非线性模型已经可以捕捉一部分预测结构；
3. 但 RF 进一步提升了表现，说明变量之间可能存在交互效应；
4. 树模型在处理水厂多变量过程关系时更有优势。
```

论文中可以这样写：

```text
The GAM model achieved a positive R² and a test RMSE close to that of XGBoost, indicating that additive nonlinear effects exist in the selected-lag feature set. However, Random Forest further reduced RMSE and achieved the highest R², suggesting that interaction effects among process variables may also contribute to NTU variation.
```

中文解释：

```text
GAM 取得了正 R²，并且 RMSE 与 XGBoost 接近，说明 selected-lag 特征中存在一定加性非线性关系。但 Random Forest 的 RMSE 更低、R² 更高，说明水厂运行变量之间可能还存在交互效应，因此 RF 更适合作为最终预测模型。
```

---

# 8. 特征重要性分析

## 8.1 Random Forest 特征重要性

Random Forest 的特征重要性如下：

| Feature | Importance |
|---|---:|
| `FILT. NTU_lag1_z` | **0.444756** |
| `CLR_lag0_z` | 0.122646 |
| `T/W FLOW_lag1_z` | 0.110712 |
| `RIVER LEVEL_lag0_z` | 0.109993 |
| `R/W FLOW_lag1_z` | 0.094995 |
| `R/W NTU_lag0_z` | 0.051505 |
| `CL2_lag0_z` | 0.040562 |
| `ALUM_lag1_z` | 0.013568 |
| `F/RIDE_lag6_z` | 0.011263 |

### 解释

最重要变量是：

```text
FILT. NTU_lag1_z
```

其重要性约为：

```text
0.444756
```

这说明：

```text
2 小时前的滤后水浊度是预测当前出厂水浊度的最关键因素。
```

这一点具有较强工艺合理性：

```text
滤后水浊度反映了前端过滤处理后的水质状态，其变化会通过后续清水池、输配或出厂环节影响当前出厂水浊度。
```

其次重要的是：

```text
CLR_lag0_z
T/W FLOW_lag1_z
RIVER LEVEL_lag0_z
R/W FLOW_lag1_z
```

这些变量分别代表：

```text
当前水质化学状态；
2 小时前处理水流量；
当前河流水位；
2 小时前原水流量。
```

说明当前模型不仅依赖水质指标，也利用了水力运行状态和外部水源状态。

---

## 8.2 XGBoost 特征重要性

XGBoost 的主要特征重要性如下：

| Feature | Importance |
|---|---:|
| `CLR_lag0_z` | **0.435604** |
| `FILT. NTU_lag1_z` | 0.296720 |
| `T/W FLOW_lag1_z` | 0.059932 |
| `F/RIDE_lag6_z` | 0.045805 |
| `RIVER LEVEL_lag0_z` | 0.040437 |
| `R/W FLOW_lag1_z` | 0.039652 |
| `CL2_lag0_z` | 0.038321 |
| `ALUM_lag1_z` | 0.022620 |
| `R/W NTU_lag0_z` | 0.020909 |

XGBoost 和 RF 的区别是：

```text
XGBoost 更强调 CLR_lag0；
RF 更强调 FILT. NTU_lag1。
```

这说明两个模型对变量贡献的理解略有差异，但共同认可：

```text
FILT. NTU_lag1 和 CLR_lag0 是核心变量。
```

---

# 9. 指定三天预测结果

题目要求预测的三个运行日为：

```text
2026-02-01
2026-02-10
2026-02-20
```

当前模型对这些日期使用 OP_DATE 进行筛选，即：

```text
一个运行日 = 当日 07:00 至次日 05:00
```

每个 OP_DATE 包含 12 个时间点。

---

## 9.1 Random Forest 三天预测汇总

| OP_DATE | n_time_points | Mean Predicted NTU | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| 2026-02-01 | 12 | **0.304056** | 0.250759 | 0.362721 | 0.038119 |
| 2026-02-10 | 12 | **0.505511** | 0.357649 | 0.610523 | 0.076593 |
| 2026-02-20 | 12 | **0.306626** | 0.264501 | 0.393629 | 0.040108 |

---

## 9.2 XGBoost 三天预测汇总

| OP_DATE | n_time_points | Mean Predicted NTU | Min | Max | Std |
|---|---:|---:|---:|---:|---:|
| 2026-02-01 | 12 | **0.296267** | 0.255531 | 0.366946 | 0.028333 |
| 2026-02-10 | 12 | **0.579046** | 0.448299 | 0.655540 | 0.068293 |
| 2026-02-20 | 12 | **0.314928** | 0.274805 | 0.388593 | 0.039651 |

---

## 9.3 三天结果解释

从 Random Forest 主模型看：

```text
2026-02-10 的平均预测 NTU 最高，为 0.505511。
```

而：

```text
2026-02-01 平均预测 NTU 为 0.304056；
2026-02-20 平均预测 NTU 为 0.306626。
```

因此可以判断：

```text
2026-02-10 是三个指定日期中出厂水浊度风险相对更高的一天。
```

但不能直接写：

```text
2026-02-10 超标。
```

除非题目明确给出 NTU 标准阈值，或者你们论文前文已经定义了超标阈值。

更稳妥的表述是：

```text
The predicted NTU on 2026-02-10 is clearly higher than those on 2026-02-01 and 2026-02-20, indicating a relatively higher turbidity risk on that operational day.
```

中文：

```text
2026-02-10 的预测 NTU 明显高于 2026-02-01 和 2026-02-20，说明该运行日存在相对更高的出厂水浊度风险。
```

---

# 10. 周期性与 drop phase 当前处理状态

根据当前图形观察，NTU 序列存在一定周期性迹象，并且局部有：

```text
短时间快速下降
```

的形状。

但是当前版本未将：

```text
OP_STEP
DAY_SIN
DAY_COS
OP_STEP one-hot
DROP_PHASE_FLAG
```

加入模型。

原因如下：

```text
1. 当前周期性并不是非常清晰稳定；
2. 快速下降阶段的位置没有被严格确定；
3. drop flag 属于人为构造变量，主观性较强；
4. 当前模型经过 clipping 后已经取得较明显提升；
5. 若贸然加入周期特征，可能增加模型复杂度和论文解释负担。
```

因此当前处理策略是：

```text
周期性和 drop phase 仅作为后续增强方向，不纳入当前主模型。
```

论文里可以写：

```text
Although the NTU series shows certain periodic patterns and occasional rapid decreases, the periodic structure is not sufficiently stable to justify explicit periodic feature construction in the current model. Therefore, periodic features and drop-phase flags were not included in this stage, but they remain potential extensions for future model improvement.
```

中文：

```text
尽管 NTU 序列表现出一定周期性以及局部快速下降现象，但该周期结构目前尚不够稳定，下降阶段也未被严格量化。因此，当前模型暂未引入显式周期特征或 drop-phase 标记，而是将其作为后续模型增强方向。
```

---

# 11. 当前模型与之前版本的对比

之前 selected-one-lag 初版结果大致为：

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Selected-one-lag Random Forest | 0.175487 | 0.268848 | 0.075639 |
| Selected-one-lag XGBoost | 0.193322 | 0.339486 | -0.473911 |

当前增强版结果为：

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Selected-one-lag Random Forest | **0.162865** | **0.216208** | **0.392095** |
| Selected-one-lag XGBoost | 0.160152 | 0.224153 | 0.346598 |
| Selected-one-lag GAM | 0.181429 | 0.224506 | 0.344542 |

可以看出：

```text
1. Random Forest 的 RMSE 从 0.268848 降至 0.216208；
2. Random Forest 的 R² 从 0.075639 提升至 0.392095；
3. XGBoost 的 R² 从负值提升为正值；
4. GAM 作为新加入模型，也取得正 R²；
5. 当前增强版整体明显优于初始 selected-one-lag 版本。
```

这一提升主要来自：

```text
1. NTU > 2 clipping 降低了极端值影响；
2. RIVER LEVEL 替换 C/W WELL LEVEL 后提供了更有效环境变量；
3. F/RIDE 缺失处理避免了样本损失；
4. GAM 的加入丰富了模型对比结构。
```

---

# 12. 当前可写进论文的核心结论

可以将当前结果总结为：

```text
基于 selected-one-lag 特征的增强模型在出厂水浊度预测中取得了较好的效果。通过对 NTU 相关变量进行上限截断处理、对 F/RIDE 缺失值进行填补，并采用时间顺序划分训练集和测试集后，Random Forest、XGBoost 和 GAM 均取得正 R²。其中，Random Forest 的 RMSE 最低、R² 最高，因此被选为最终主模型。
```

英文版本：

```text
The enhanced selected-one-lag modeling framework achieved improved predictive performance for output water turbidity. After applying upper-bound clipping to NTU-related variables, filling missing F/RIDE values with zero, and adopting a time-based train-test split, all three models obtained positive R² values. Among them, Random Forest achieved the lowest RMSE and the highest R², and was therefore selected as the final predictive model.
```

---

# 13. 当前报告写法建议

在论文中建议这样组织第一题模型部分：

```text
1. 数据预处理
   - OP_DATE 构造
   - F/RIDE 缺失填 0
   - NTU > 2 上限截断

2. 滞后特征选择
   - 每个变量给定候选 lag
   - 根据训练集相关性选择一个 best lag
   - 构造 selected-one-lag 特征集

3. 模型方法
   - Random Forest
   - XGBoost
   - GAM

4. 评价方法
   - 时间顺序 train/test split
   - MAE / RMSE / R² / MAPE

5. 模型结果
   - RF 最优
   - GAM 作为可解释非线性基准
   - XGBoost 作为 boosting 对比模型

6. 特征重要性
   - FILT. NTU_lag1 最重要
   - CLR_lag0、FLOW、RIVER LEVEL 也有贡献

7. 三天预测结果
   - 2026-02-10 风险最高
   - 2026-02-01 与 2026-02-20 较接近

8. 周期性说明
   - 当前观察到周期迹象
   - 暂未加入周期特征
   - 后续可作为增强方向
```

---

# 14. 当前阶段仍需注意的问题

## 14.1 Clipping 需要解释清楚

如果最终采用 clipping，必须说明：

```text
NTU > 2 被映射为 2。
```

这不是错误，但必须透明。

否则评审可能认为你改变了目标变量却没有说明。

---

## 14.2 MAPE 不宜作为主要指标

当前 MAPE 仍然较高：

```text
RF MAPE = 48.743367%
XGBoost MAPE = 45.401629%
GAM MAPE = 51.885268%
```

原因是 NTU 数值本身较小，接近 0 时 MAPE 会被放大。

因此论文中建议主要讨论：

```text
MAE
RMSE
R²
```

MAPE 可以放入表格，但不要作为主要结论依据。

---

## 14.3 GAM 没有直接 feature_importances_

当前 feature importance 主要来自：

```text
Random Forest
XGBoost
```

GAM 不直接输出与树模型相同形式的 feature importance。

因此论文可以写：

```text
Feature importance was mainly interpreted using the tree-based models, while GAM was used as an interpretable nonlinear benchmark for performance comparison.
```

---

## 14.4 周期性暂不进入主模型是合理的

虽然图中可见一定周期结构，但如果没有严格验证，直接加入 drop flag 可能过度主观。

当前更稳妥的表述是：

```text
周期性特征作为后续工作，而非当前主模型组成部分。
```

---

# 15. 最终推荐结论

当前第一题最推荐的最终方案是：

```text
Enhanced Selected-one-lag Random Forest
```

其主要组成是：

```text
selected-one-lag 特征
+ NTU clipping
+ F/RIDE 缺失填 0
+ RIVER LEVEL 环境变量
+ 训练集标准化
+ median imputation
+ 时间顺序 train/test split
```

最终主模型指标：

| Metric | Value |
|---|---:|
| MAE | 0.162865 |
| RMSE | 0.216208 |
| R² | 0.392095 |
| MAPE | 48.743367% |

最终三天预测结论：

```text
2026-02-10 的预测 NTU 最高，说明该日相对浊度风险最高；
2026-02-01 与 2026-02-20 的预测水平较接近，均明显低于 2026-02-10。
```

当前不建议继续大幅改模型结构。后续如果时间允许，可以做一个补充实验：

```text
Enhanced Selected-one-lag RF + periodic features
```

但不建议在没有稳定证据前直接把周期性和 drop flag 写成主模型特征。

---

# 16. 可直接用于论文的简洁版结论

```text
After constructing the selected-one-lag feature set, an enhanced modeling framework was developed by applying upper-bound clipping to NTU-related variables and filling missing F/RIDE values with zero. Three models, including Random Forest, XGBoost, and a spline-based GAM, were compared under a time-based train-test split. The Random Forest model achieved the best overall performance, with MAE = 0.162865, RMSE = 0.216208, and R² = 0.392095. Feature importance analysis showed that FILT. NTU_lag1 was the dominant predictor, followed by CLR_lag0, T/W FLOW_lag1, RIVER LEVEL_lag0, and R/W FLOW_lag1. For the three specified operational dates, the predicted NTU on 2026-02-10 was substantially higher than those on 2026-02-01 and 2026-02-20, indicating a relatively higher turbidity risk on that day.
```

中文：

```text
在构建 selected-one-lag 特征集后，本文进一步通过 NTU 上限截断和 F/RIDE 缺失值填补构建增强型预测框架，并比较了 Random Forest、XGBoost 和基于样条的 GAM 三类模型。在时间顺序划分的测试集上，Random Forest 取得最佳整体表现，MAE 为 0.162865，RMSE 为 0.216208，R² 为 0.392095。特征重要性结果表明，FILT. NTU_lag1 是最主要预测因子，其次为 CLR_lag0、T/W FLOW_lag1、RIVER LEVEL_lag0 和 R/W FLOW_lag1。对于题目指定的三个运行日，2026-02-10 的预测 NTU 明显高于 2026-02-01 和 2026-02-20，说明该日存在相对更高的出厂水浊度风险。
```
