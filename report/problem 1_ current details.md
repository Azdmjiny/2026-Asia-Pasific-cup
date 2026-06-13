# Problem 1 当前最有用详细报告：出厂水浊度 NTU 的影响因素、滞后特征与预测模型

## 0. 当前结论总览

目前 Problem 1 已经形成了一条比较完整的建模链路：

```text
merged.xlsx
→ 数据审计与 baseline 建模
→ 同期相关性分析
→ 滞后相关性分析
→ NTU 自相关与周期性分析
→ selected-one-lag 特征选择
→ RF / XGBoost 预测模型
→ 三个指定运行日 NTU 预测
```

当前最推荐的主模型是：

```text
Selected-one-lag Random Forest
```

原因如下：

| 指标 | Selected-one-lag RF | Selected-one-lag XGBoost |
|---|---:|---:|
| MAE | **0.175487** | 0.193322 |
| RMSE | **0.268848** | 0.339486 |
| R² | **0.075639** | -0.473911 |
| MAPE | 51.657555% | 50.571126% |

核心判断：

1. **Selected-one-lag RF 明显优于当前 full-sample baseline。**
2. RF 的 R² 已经从 baseline 的负值提升为正值，说明 selected-lag 特征确实改善了模型预测能力。
3. 但是 RF 的 R² 只有约 0.076，说明模型只是“有效改善”，不能写成“预测效果很好”。
4. XGBoost 在当前 selected-one-lag 数据上仍为负 R²，不建议作为主模型。
5. 特征重要性显示，`FILT. NTU_lag1_z` 是当前最核心变量，其次为 `CLR_lag0_z` 和 `RIVER LEVEL_lag0_z`。
6. 图形观察显示 NTU 序列存在一定“阶段性快速下降”形态，后续最值得加入的是**日内周期特征**，例如 `OP_STEP`、`DAY_SIN`、`DAY_COS`、`OP_STEP` one-hot。

论文中建议使用的谨慎表述：

```text
Compared with the synchronous baseline model, the selected-one-lag Random Forest reduced the prediction error and achieved a positive R², indicating that lagged process variables improve the predictive ability for output water turbidity. However, the relatively small R² suggests that the model still has limited explanatory power and should be interpreted as a practical predictive baseline rather than a highly accurate deterministic model.
```

中文：

```text
与同期变量 baseline 模型相比，selected-one-lag Random Forest 明显降低了预测误差，并使 R² 从负值提升为正值，说明滞后工艺变量能够改善出厂水浊度的预测能力。但由于 R² 仍然较低，该模型应被解释为具有实际改进效果的预测基线，而不是高精度确定性模型。
```
---
# 1. Baseline 建模结果回顾

`p1.ipynb` 中做了两种 baseline 方案。

## 1.1 方案 A：删除缺失特征

方案 A 的处理方式：

```text
删除缺失严重的输入特征，但保留全部 NTU 有效样本。
```

样本情况：

| 项目 | 数值 |
|---|---:|
| 样本数 | 5124 |
| 输入特征数 | 8 |
| 训练集 | 4099 |
| 测试集 | 1025 |

使用特征：

```text
RIVER LEVEL
R/W FLOW
R/W NTU
R/W CLR
FILT. NTU
C/W WELL LEVEL
CLR
T/W FLOW
```

模型结果：

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | 0.2199 | 0.3864 | -0.9091 |
| Random Forest | 0.2194 | 0.4130 | -1.1817 |

解释：

1. 方案 A 保留了较大的样本量，因此更接近真实完整时间序列。
2. 但是模型 R² 均为负数，说明只使用同期变量时，对测试集 NTU 波动解释不足。
3. 方案 A 最佳模型为 XGBoost，但其 R² 仍为 -0.9091，不能作为最终强模型。

---

## 1.2 方案 B：删除缺失行

方案 B 的处理方式：

```text
保留更多输入特征，但删除任何输入变量缺失的行。
```

样本情况：

| 项目 | 数值 |
|---|---:|
| 样本数 | 1049 |
| 输入特征数 | 15 |
| 训练集 | 839 |
| 测试集 | 210 |

模型结果：

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 0.0660 | 0.0921 | 0.5875 |
| XGBoost | 0.0769 | 0.0976 | 0.5364 |

解释：

1. 方案 B 的数值结果明显更好。
2. 但是方案 B 只保留 1049 行，删除了大量含缺失值的数据。
3. 测试区间也显著缩短，且缺失行删除可能造成样本选择偏差。
4. 因此，方案 B 更适合作为参考，不建议作为最终主模型依据。
5. 最终主模型应优先使用保留较完整时间序列的 selected-lag 方案。

---

## 1.3 Baseline 变量重要性与相关性结论

方案 A 中，模型重要性和相关性反复指向：

```text
FILT. NTU
CLR
RIVER LEVEL
R/W FLOW
T/W FLOW
```

方案 A 的 Pearson 相关性前几项：

| 变量 | Pearson corr |
|---|---:|
| FILT. NTU | 0.6947 |
| CLR | -0.4827 |
| R/W FLOW | 0.1409 |
| T/W FLOW | 0.1183 |
| C/W WELL LEVEL | -0.0783 |
| RIVER LEVEL | 0.0596 |

方案 A 的 Spearman 相关性前几项：

| 变量 | Spearman corr |
|---|---:|
| R/W FLOW | 0.3785 |
| T/W FLOW | 0.3523 |
| FILT. NTU | 0.2297 |
| R/W CLR | -0.1690 |
| R/W NTU | -0.1570 |

关键解释：

1. `FILT. NTU` 与当前 `NTU` 的线性相关性最强。
2. `R/W FLOW` 和 `T/W FLOW` 的 Spearman 明显高于 Pearson，说明流量变量与 NTU 之间可能存在非线性或单调关系。
3. `CLR` 与 NTU 呈明显负相关，但不能直接解释为因果关系。
4. 这些结果为后续 selected-lag 特征选择提供了依据。

---

# 1.4 滞后相关性分析

`p1_1.ipynb` 中对候选变量计算了：

```text
corr(X_{t-k}, NTU_t), k = 0, 1, ..., 12
```

由于数据每 2 小时一条：

```text
lag0  = 当前时刻
lag1  = 2 小时前
lag2  = 4 小时前
...
lag12 = 24 小时前
```

---

## 2.1 候选变量

候选变量包括：

```text
R/W NTU
FILT. NTU
R/W FLOW
T/W FLOW
C/W WELL LEVEL
ALUM
R/W PH
PH
CLR
CL2
RIVER LEVEL
R/W CLR
```

候选变量缺失率情况：

| 变量 | 缺失率 |
|---|---:|
| CL2 | 31.26% |
| ALUM | 30.11% |
| R/W PH | 30.11% |
| PH | 30.11% |
| R/W NTU | 0.00% |
| FILT. NTU | 0.00% |
| R/W FLOW | 0.00% |
| T/W FLOW | 0.00% |
| C/W WELL LEVEL | 0.00% |
| CLR | 0.00% |
| RIVER LEVEL | 0.00% |
| R/W CLR | 0.00% |

---

## 2.2 各变量最强滞后结果

`best_lag_summary.csv` 的主要结果如下：

| 变量 | Pearson 最优 lag | Pearson 小时 | Pearson corr | Spearman 最优 lag | Spearman 小时 | Spearman corr |
|---|---:|---:|---:|---:|---:|---:|
| FILT. NTU | 0 | 0h | 0.694743 | 1 | 2h | 0.232051 |
| CLR | 0 | 0h | -0.482685 | 0 | 0h | -0.109983 |
| R/W FLOW | 1 | 2h | 0.144300 | 1 | 2h | 0.379603 |
| T/W FLOW | 3 | 6h | 0.122386 | 1 | 2h | 0.355191 |
| RIVER LEVEL | 12 | 24h | 0.116780 | 0 | 0h | -0.134295 |
| CL2 | 0 | 0h | -0.135404 | 1 | 2h | -0.145098 |
| R/W NTU | 12 | 24h | 0.050883 | 0 | 0h | -0.157045 |
| ALUM | 11 | 22h | 0.040155 | 1 | 2h | -0.030688 |
| C/W WELL LEVEL | 0 | 0h | -0.078290 | 12 | 24h | 0.101641 |
| PH | 12 | 24h | 0.050013 | 12 | 24h | 0.177107 |
| R/W PH | 4 | 8h | -0.091544 | 12 | 24h | -0.216993 |
| R/W CLR | 12 | 24h | 0.036846 | 0 | 0h | -0.169043 |

---

## 2.3 滞后相关性解释

### 2.3.1 FILT. NTU

`FILT. NTU` 是最稳定、最核心的变量。

```text
Pearson 最强：lag0，corr = 0.694743
Spearman 最强：lag1，corr = 0.232051
```

解释：

1. 滤后水浊度与出厂水浊度具有直接水质联系。
2. `lag1` 也保留较强关系，说明 2 小时前的滤后水状态对当前出厂水浊度有预测价值。
3. 这也是当前最终模型中 `FILT. NTU_lag1_z` 成为最重要特征的原因。

---

### 2.3.2 CLR

```text
Pearson 最强：lag0，corr = -0.482685
```

解释：

1. `CLR` 与 `NTU` 存在较强同期负相关。
2. 但该关系只能解释为统计关联，不能直接写成因果。
3. 当前最终模型保留 `CLR_lag0_z`，并且它在 RF 中是第二重要特征。

---

### 2.3.3 R/W FLOW 与 T/W FLOW

```text
R/W FLOW Spearman 最强：lag1，corr = 0.379603
T/W FLOW Spearman 最强：lag1，corr = 0.355191
```

解释：

1. 流量变量的 Spearman 相关性明显强于 Pearson。
2. 这说明流量与 NTU 之间可能不是简单线性关系，而是非线性或单调关系。
3. 使用 RF / XGBoost 这类非线性模型是合理的。
4. 当前最终模型保留 `R/W FLOW_lag1_z` 和 `T/W FLOW_lag1_z`。

---

### 2.3.4 RIVER LEVEL

```text
Pearson 最强：lag12，corr = 0.116780
Spearman 最强：lag0，corr = -0.134295
```

解释：

1. 河流水位与 NTU 的直接相关性不强，但它反映原水侧环境状态。
2. 当前建模中已经用 `RIVER LEVEL` 替换 `C/W WELL LEVEL`。
3. 最终 selected-one-lag 模型保留 `RIVER LEVEL_lag0_z`。
4. 在 RF 特征重要性中，`RIVER LEVEL_lag0_z` 排名第三，说明它作为环境背景变量是有贡献的。

---

### 2.3.5 ALUM 与 F/RIDE

`ALUM` 的简单相关性较弱：

```text
Pearson 最强 corr = 0.040155
Spearman 最强 corr = -0.030688
```

`F/RIDE` 在最终建模中使用 `lag6`，并且缺失值已经设置为 0 或在建模前补 0。

解释：

1. `ALUM` 和 `F/RIDE` 都不适合作为核心预测变量。
2. 它们可以作为辅助化学指标保留。
3. 从当前模型结果看，二者特征重要性都较低。

---

# 3. NTU 自相关性分析

`p1_2.ipynb` 用于分析目标变量 `NTU` 自身的时间连续性：

```text
corr(NTU_{t-k}, NTU_t), k = 0,1,...,12
```

主要结果：

| lag | 小时 | Pearson | Spearman |
|---:|---:|---:|---:|
| lag1 | 2h | 0.881395 | 0.969947 |
| lag2 | 4h | 0.790315 | 0.944204 |
| lag3 | 6h | 0.724175 | 0.921659 |
| lag6 | 12h | 0.588330 | 0.871135 |
| lag12 | 24h | 0.460812 | 约 0.805 |

解释：

1. NTU 存在非常强的短期时间连续性。
2. 2 小时前的 NTU 与当前 NTU 高度相关。
3. 即使 24 小时前的 NTU 仍然有一定相关性。
4. 这说明 NTU 序列本身具有明显“惯性”。

---

## 3.1 为什么不直接把 NTU_lag 放入主模型

虽然 `NTU_lag1` 很强，但不建议作为第一版主模型输入。

原因是三个目标日期的 NTU 本身缺失：

| 检查项 | 缺失数 |
|---|---:|
| NTU | 36 |
| NTU_lag1 | 35 |
| NTU_lag2 | 34 |
| NTU_lag3 | 33 |
| NTU_lag6 | 30 |
| NTU_lag12 | 24 |

如果模型使用 `NTU_lag1`，那么目标日期后续时间点会依赖前一个时间点的预测值：

```text
预测 NTU_09:00 需要 NTU_07:00
但 NTU_07:00 也是预测出来的
```

这会形成递推预测：

```text
prediction → next prediction → next prediction
```

风险是：

```text
误差会逐步累积。
```

因此当前报告建议：

```text
NTU 自相关性用于说明时间惯性，不作为第一版主模型的直接输入。
```

---

## 5.3 推荐加入的周期性特征

水厂一个运行日有 12 个时间点：

| 时间 | OP_STEP |
|---|---:|
| 07:00 | 0 |
| 09:00 | 1 |
| 11:00 | 2 |
| 13:00 | 3 |
| 15:00 | 4 |
| 17:00 | 5 |
| 19:00 | 6 |
| 21:00 | 7 |
| 23:00 | 8 |
| 01:00 | 9 |
| 03:00 | 10 |
| 05:00 | 11 |

建议构造：

```python
hour_to_step = {
    7: 0,
    9: 1,
    11: 2,
    13: 3,
    15: 4,
    17: 5,
    19: 6,
    21: 7,
    23: 8,
    1: 9,
    3: 10,
    5: 11,
}

df["OP_STEP"] = df["DATETIME"].dt.hour.map(hour_to_step)

df["DAY_SIN"] = np.sin(2 * np.pi * df["OP_STEP"] / 12)
df["DAY_COS"] = np.cos(2 * np.pi * df["OP_STEP"] / 12)

step_dummies = pd.get_dummies(df["OP_STEP"], prefix="OP_STEP", drop_first=True)
df = pd.concat([df, step_dummies], axis=1)
```

推荐先加入：

```text
DAY_SIN
DAY_COS
OP_STEP_1 到 OP_STEP_11
```

暂时不建议直接加入 `DROP_PHASE_FLAG`。

原因：

1. `DROP_PHASE_FLAG` 是人工判断特征。
2. 如果没有足够证据，论文里容易被问为什么这些时段被定义为下降阶段。
3. 更稳妥的是先用 `OP_STEP one-hot` 让模型自动学习哪些时段更重要。
4. 如果后续发现 `OP_STEP_8`、`OP_STEP_9`、`OP_STEP_10` 很重要，再合成 `DROP_PHASE_FLAG`。

---

# 6. 当前 selected-one-lag 建模数据

`p1_3.ipynb` 当前使用的文件为：

```text
data/selected_one_lag_model_data.xlsx
```

数据规模：

| 项目 | 数值 |
|---|---:|
| 行数 | 5460 |
| 列数 | 12 |
| 输入特征数 | 9 |
| 目标变量 | target_NTU |

当前建模特征：

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

解释：

| 特征 | 含义 |
|---|---|
| FILT. NTU_lag1_z | 2 小时前标准化滤后水浊度 |
| CLR_lag0_z | 当前标准化 CLR |
| R/W FLOW_lag1_z | 2 小时前标准化原水流量 |
| T/W FLOW_lag1_z | 2 小时前标准化出厂水流量 |
| RIVER LEVEL_lag0_z | 当前标准化河流水位 |
| R/W NTU_lag0_z | 当前原水浊度 |
| CL2_lag0_z | 当前余氯相关指标 |
| ALUM_lag1_z | 2 小时前明矾/混凝剂投加量 |
| F/RIDE_lag6_z | 12 小时前 F/RIDE |

其中：

```text
C/W WELL LEVEL 已被替换为 RIVER LEVEL。
PUMP DUTY 不进入当前主模型。
F/RIDE 缺失值在建模阶段填为 0。
```

---

## 6.1 selected-one-lag 特征选择逻辑

当前策略不是把所有滞后项都放入模型，而是：

```text
每个原始变量只保留一个最重要 lag。
```

选择逻辑：

```text
1. 为每个变量设置候选 lag。
2. 在训练段中计算各候选 lag 与当前 target_NTU 的相关性。
3. 选择绝对相关性最高的那个 lag。
4. 每个变量最终只保留一个特征。
```

数学表达：

```text
best_lag(X) = argmax_k |Corr(X_{t-k}, NTU_t)|
```

这样做的优点：

1. 减少特征冗余。
2. 降低过拟合风险。
3. 提高论文解释清晰度。
4. 避免同一变量的多个高度相关 lag 同时进入模型。

---

# 7. 当前 selected-one-lag 模型训练结果

## 7.1 训练/测试划分

当前 `p1_3.ipynb` 使用时间顺序划分：

| 项目 | 数值 |
|---|---:|
| 训练集 | 4099 |
| 测试集 | 1025 |
| 训练时间范围 | 2025-01-01 07:00 至 2025-12-08 19:00 |
| 测试时间范围 | 2025-12-08 21:00 至 2026-04-01 05:00 |

处理流程：

```text
1. 按时间排序；
2. 排除三个 target OP_DATE；
3. 前 80% 作为训练集；
4. 后 20% 作为测试集；
5. 输入特征做训练集统计量标准化；
6. 其他缺失值用 median imputation；
7. target_NTU 不标准化。
```

---

## 7.2 模型结果

| 模型 | 训练样本 | 测试样本 | 特征数 | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected-one-lag Random Forest | 4099 | 1025 | 9 | **0.175487** | **0.268848** | **0.075639** | 51.657555% |
| Selected-one-lag XGBoost | 4099 | 1025 | 9 | 0.193322 | 0.339486 | -0.473911 | 50.571126% |

结论：

```text
Selected-one-lag Random Forest 是当前最佳模型。
```

原因：

1. MAE 最低。
2. RMSE 最低。
3. R² 唯一为正。
4. 相比 baseline 的同期变量模型有明显提升。
5. 对当前数据的非线性关系更稳定。

---

## 7.3 与 baseline 对比

与方案 A baseline 对比：

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Baseline XGBoost | 0.2199 | 0.3864 | -0.9091 |
| Baseline Random Forest | 0.2194 | 0.4130 | -1.1817 |
| Selected-one-lag RF | **0.1755** | **0.2688** | **0.0756** |

解释：

1. Selected-one-lag RF 的 RMSE 从 baseline RF 的 0.4130 降至 0.2688。
2. 相比 baseline XGBoost 的 0.3864，也有明显降低。
3. R² 从负值提升为正值。
4. 这说明 selected-lag 特征比简单同期变量更适合当前任务。

注意：

```text
不要拿方案 B 的 RMSE = 0.0921 直接与 selected-one-lag RF 比较。
```

因为方案 B 删除了大量含缺失值行，只剩 1049 条样本，测试集也明显不同，不是同一评价条件。

---

# 8. 当前模型特征重要性

## 8.1 Random Forest 特征重要性

| 排名 | 特征 | 重要性 |
|---:|---|---:|
| 1 | FILT. NTU_lag1_z | 0.457392 |
| 2 | CLR_lag0_z | 0.236401 |
| 3 | RIVER LEVEL_lag0_z | 0.079347 |
| 4 | T/W FLOW_lag1_z | 0.063453 |
| 5 | CL2_lag0_z | 0.057825 |
| 6 | R/W FLOW_lag1_z | 0.056032 |
| 7 | R/W NTU_lag0_z | 0.035854 |
| 8 | F/RIDE_lag6_z | 0.006873 |
| 9 | ALUM_lag1_z | 0.006822 |

解释：

1. `FILT. NTU_lag1_z` 的重要性接近 0.46，是绝对核心变量。
2. `CLR_lag0_z` 是第二重要变量。
3. `RIVER LEVEL_lag0_z` 排名第三，说明用 RIVER LEVEL 替代 C/W WELL LEVEL 是有价值的。
4. 流量变量 `T/W FLOW_lag1_z` 和 `R/W FLOW_lag1_z` 有辅助贡献。
5. `F/RIDE_lag6_z` 和 `ALUM_lag1_z` 当前贡献非常低，可作为辅助变量保留，也可在下一版模型中做消融测试。

---

## 8.2 XGBoost 特征重要性

| 排名 | 特征 | 重要性 |
|---:|---|---:|
| 1 | CLR_lag0_z | 0.611275 |
| 2 | FILT. NTU_lag1_z | 0.182955 |
| 3 | CL2_lag0_z | 0.042712 |
| 4 | F/RIDE_lag6_z | 0.040796 |
| 5 | T/W FLOW_lag1_z | 0.035614 |
| 6 | R/W FLOW_lag1_z | 0.033653 |
| 7 | RIVER LEVEL_lag0_z | 0.027887 |
| 8 | R/W NTU_lag0_z | 0.013635 |
| 9 | ALUM_lag1_z | 0.011472 |

解释：

1. XGBoost 过度依赖 `CLR_lag0_z`。
2. XGBoost 测试集 R² 为负，说明其当前预测稳定性不如 RF。
3. 论文中可以展示 XGBoost 作为对比模型，但不建议作为最终预测模型。

---

# 9. 三个指定运行日预测结果

最终模型对三个 OP_DATE 进行了 12 个时间点预测。

## 9.1 OP_DATE 日均预测结果

### Random Forest

| OP_DATE | 时间点数 | 平均预测 NTU | 最小值 | 最大值 | 标准差 |
|---|---:|---:|---:|---:|---:|
| 2026-02-01 | 12 | 0.304217 | 0.250788 | 0.360998 | 0.037200 |
| 2026-02-10 | 12 | 0.502998 | 0.351554 | 0.598891 | 0.078125 |
| 2026-02-20 | 12 | 0.300342 | 0.265883 | 0.386391 | 0.041728 |

### XGBoost

| OP_DATE | 时间点数 | 平均预测 NTU | 最小值 | 最大值 | 标准差 |
|---|---:|---:|---:|---:|---:|
| 2026-02-01 | 12 | 0.287732 | 0.245668 | 0.355208 | 0.026624 |
| 2026-02-10 | 12 | 0.566080 | 0.458050 | 0.689488 | 0.070920 |
| 2026-02-20 | 12 | 0.319021 | 0.279915 | 0.404967 | 0.046585 |

由于 RF 是当前主模型，论文建议主要报告 RF 预测结果。

---

## 9.2 预测日期解释

根据 Random Forest：

```text
2026-02-10 的预测平均 NTU 最高，约为 0.5030。
2026-02-01 和 2026-02-20 的预测平均 NTU 接近，约为 0.30。
```

解释：

1. 2026-02-10 的出厂水浊度风险相对更高。
2. 2026-02-01 和 2026-02-20 预测水平较低且更稳定。
3. 但不能直接写“超标”，除非题目或标准明确给出了超标阈值。
4. 如果论文需要描述风险，可写“相对浊度水平较高”或“需要重点关注”。

推荐论文表述：

```text
Among the three target operating dates, 2026-02-10 has the highest predicted mean NTU, suggesting a relatively higher turbidity level on that day. In contrast, 2026-02-01 and 2026-02-20 show lower and more stable predicted NTU values.
```

中文：

```text
在三个指定运行日中，2026-02-10 的预测平均 NTU 最高，说明该日出厂水浊度水平相对较高，需要重点关注。相比之下，2026-02-01 和 2026-02-20 的预测 NTU 较低且波动较小。
```

---

# 10. 当前模型的主要优点

## 10.1 特征选择逻辑清晰

当前模型不是盲目堆叠变量，而是基于：

```text
滞后相关性分析
→ 每个变量选择一个最重要 lag
→ 构造 selected-one-lag 特征集
```

这使模型具备较好的可解释性。

---

## 10.2 比同期 baseline 明显提升

Selected-one-lag RF 的 RMSE 明显低于方案 A baseline：

```text
0.4130 → 0.2688
```

这说明滞后特征确实有效。

---

## 10.3 特征重要性结果符合水处理逻辑

最重要变量是：

```text
FILT. NTU_lag1_z
```

其含义为：

```text
2 小时前的滤后水浊度
```

这符合水处理流程中“滤后水状态影响后续出厂水状态”的直觉。

---

## 10.4 已经处理了指定日期缺失问题

当前模型没有直接使用 `NTU_lag`，避免了目标日期递推预测问题。

这是一个很重要的稳健性设计。

---

# 11. 当前模型的局限性

## 11.1 R² 仍然较低

当前最佳 RF：

```text
R² = 0.075639
```

说明模型解释能力有限。论文中应避免夸大。

建议写：

```text
The selected-lag model improves the prediction error compared with the baseline, but the relatively low R² indicates that additional dynamic or operational factors may still be missing.
```

---

## 11.2 MAPE 较高，不宜作为核心指标

当前 MAPE 约为：

```text
RF: 51.66%
XGBoost: 50.57%
```

由于 NTU 数值较小，MAPE 容易被接近 0 的目标值放大。因此论文应主要使用：

```text
MAE
RMSE
R²
```

MAPE 可作为补充，不建议重点讨论。

---

## 11.3 当前周期性还没有进入模型

图形上已经观察到：

```text
周期性快速下降现象
```

但当前 `p1_3.ipynb` 中还没有加入：

```text
OP_STEP
DAY_SIN
DAY_COS
OP_STEP one-hot
```

因此下一步最值得做的是：

```text
selected-one-lag + periodic features
```

---

## 11.4 当前输入可能存在重复标准化问题

当前建模文件中的列名已经带有：

```text
_lag1_z
_lag0_z
```

而训练脚本中又执行了训练集标准化。

对 Random Forest 和 XGBoost 来说，线性缩放一般不会显著影响结果；但从代码规范和论文表达上，建议后续统一为一种标准化方式：

建议方案：

```text
原始 selected-one-lag 特征不带 _z
训练脚本中统一使用训练集均值和标准差做标准化
```

或：

```text
如果输入文件已标准化，训练脚本不再重复标准化。
```

论文中不要写“标准化了两次”，统一表述为：

```text
Input features were standardized using training-set statistics before model training.
```

---

# 13. 建议最终论文写法结构

Problem 1 可以按如下结构写：

## 13.1 数据预处理

写：

```text
首先将 DATE 和 TIME 合并为 DATETIME，并按照时间顺序排序。考虑到水厂运行日从 07:00 开始，到次日 05:00 结束，本文进一步构造 OP_DATE，用于统一表示水厂运行日。对于目标变量 NTU 缺失的行，在模型训练阶段不参与监督学习；对于输入变量缺失值，则在训练集中采用中位数插补。
```

---

## 13.2 Baseline 建模

写：

```text
首先构建同期变量 baseline 模型，即使用当前时刻输入变量预测当前 NTU。结果显示，baseline 模型在测试集上的 R² 为负，说明仅使用同期变量难以解释出厂水浊度的动态变化。因此，有必要进一步考虑滞后效应。
```

---

## 13.3 滞后相关性分析

写：

```text
为刻画水处理过程中的时间延迟效应，本文对候选变量构造 lag0 至 lag12 的滞后项，其中每一阶 lag 对应 2 小时。随后计算各滞后项与当前 NTU 的 Pearson 和 Spearman 相关系数。结果显示，FILT. NTU 与 NTU 的相关性最强，R/W FLOW 和 T/W FLOW 的 Spearman 相关性明显高于 Pearson，说明流量变量可能与 NTU 存在非线性单调关系。
```

---

## 13.4 selected-one-lag 特征选择

写：

```text
为避免同一变量多个滞后项同时进入模型导致特征冗余，本文采用 selected-one-lag 策略。对于每个候选变量，首先给定若干候选滞后阶数，然后在训练集上计算各候选滞后项与当前 NTU 的绝对相关性，仅保留相关性最高的一个滞后项作为最终输入特征。
```

---

## 13.5 模型训练与比较

写：

```text
基于 selected-one-lag 特征集，本文分别训练 Random Forest 和 XGBoost 模型。训练集和测试集按照时间顺序划分，避免随机划分造成时间信息泄漏。输入变量使用训练集统计量进行标准化，剩余缺失值采用中位数插补，目标变量 NTU 保持原始量纲。
```

然后放表：

| 模型 | MAE | RMSE | R² |
|---|---:|---:|---:|
| Selected-one-lag Random Forest | 0.175487 | 0.268848 | 0.075639 |
| Selected-one-lag XGBoost | 0.193322 | 0.339486 | -0.473911 |

结论：

```text
Random Forest 在 MAE、RMSE 和 R² 上均优于 XGBoost，因此选为最终预测模型。
```

---

## 13.6 指定日期预测

放表：

| OP_DATE | 平均预测 NTU | 最小值 | 最大值 | 标准差 |
|---|---:|---:|---:|---:|
| 2026-02-01 | 0.304217 | 0.250788 | 0.360998 | 0.037200 |
| 2026-02-10 | 0.502998 | 0.351554 | 0.598891 | 0.078125 |
| 2026-02-20 | 0.300342 | 0.265883 | 0.386391 | 0.041728 |

写：

```text
2026-02-10 的预测 NTU 明显高于另外两个运行日，说明该日出厂水浊度水平相对较高，应作为重点关注日期。
```

---

# 14. 当前最应该交付的文件

建议交给论文手以下文件：

```text
outputs/problem1/selected_one_lag_model_results.xlsx
outputs/problem1/selected_one_lag_feature_importance.xlsx
outputs/problem1/target_dates_predictions.xlsx
outputs/problem1/target_dates_prediction_summary.xlsx
outputs/problem1/selected_one_lag_test_predictions.xlsx
outputs/problem1/figures/
```

建议交给建模手继续改进的文件：

```text
data/selected_one_lag_model_data.xlsx
codes/p1_3.ipynb
```

建议下一版新建：

```text
p1_4_periodic_selected_lag_model.ipynb
```

用于测试：

```text
selected-one-lag + periodic features
```

---

# 15. 当前最终建议

## 15.1 当前主模型

```text
Selected-one-lag Random Forest
```

## 15.2 当前主结论

```text
滞后特征显著改善 baseline 表现，尤其是 FILT. NTU_lag1 对当前 NTU 具有最高预测贡献。
```

## 15.3 当前不能夸大的地方

不要写：

```text
模型预测效果很好。
模型能够准确预测 NTU。
模型已经完全捕捉 NTU 周期。
```

应该写：

```text
模型相比 baseline 有明显改进，但解释能力仍有限。
```