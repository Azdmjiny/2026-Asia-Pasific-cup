# 第三题：基于公式分项特征的 GRU 模型设计

## 1. 建模目标

第三题预测对象应为出厂水浊度 `NTU`，而不是 `FILT. NTU`。

数据每 2 小时采样一次，因此将未来 1--12 小时预测离散为：

$$
2h,\ 4h,\ 6h,\ 8h,\ 10h,\ 12h
$$

推荐预测相对当前时刻的 NTU 增量：

$$
\mathbf y_t=
\left[
C_{t+1}-C_t,\,
C_{t+2}-C_t,\,
\ldots,\,
C_{t+6}-C_t
\right]
$$

模型预测后还原未来 NTU：

$$
\hat C_{t+h}=C_t+\widehat{\Delta C}_{t+h}
$$

这样可以避免模型只依靠 `NTU` 的强自相关获得较高指标。

## 2. 公式分项特征

参考 `p3_inv2.ipynb` 中的物料衡算：

$$
C_{t+1}
=
\underbrace{C_t}_{\mathrm{current\ state}}
+
\underbrace{\frac{\Delta t\,Q_{in,t}C_{in,t}}{V_t}}_{\mathrm{inflow}}
-
\underbrace{\frac{\Delta t\,Q_{out,t}C_t}{V_t}}_{\mathrm{outflow}}
$$

其中：

- $\Delta t=2$ 小时；
- $C_t$：当前出厂水浊度 `NTU`；
- $C_{in,t}$：滤后水浊度 `FILT. NTU`；
- $V_t$：使用 `C/W WELL LEVEL` 作为相对体积代理；
- $Q_{out,t}$：使用 `T/W FLOW`；
- $Q_{in,t}$：分别测试 `R/W FLOW` 和 `T/W FLOW`。

构造以下分项特征。

### 2.1 当前状态项

$$
S_t=C_t
$$

对应：

```text
current_NTU = NTU
```

### 2.2 入流项

$$
I_t=\frac{2Q_{in,t}C_{in,t}}{L_t}
$$

方案 A：

$$
I_t^{RW}
=
\frac{2\,Q_{in,t}^{RW}C_{in,t}}{L_t}
$$

其中 $Q_{in,t}^{RW}$ 对应 `R/W FLOW`，$C_{in,t}$ 对应 `FILT. NTU`，$L_t$ 对应 `C/W WELL LEVEL`。

方案 B：

$$
I_t^{TW}
=
\frac{2\,Q_{in,t}^{TW}C_{in,t}}{L_t}
$$

其中 $Q_{in,t}^{TW}$ 对应 `T/W FLOW`。

### 2.3 出流项

$$
O_t=
\frac{2Q_{out,t}C_t}{L_t}
$$

即：

$$
O_t=
\frac{2\,Q_{out,t}C_t}{L_t}
$$

其中 $Q_{out,t}$ 对应 `T/W FLOW`，$C_t$ 对应 `NTU`。

出流项在数据中保存为正值，其负向作用交给模型学习。

## 3. GRU 输入特征

推荐每个时刻使用以下特征：

| 特征 | 含义 |
|---|---|
| `current_NTU` | 当前出厂水浊度 |
| `inflow_term` | 公式入流项 |
| `outflow_term` | 公式出流项 |
| `HRT_PROXY` | 水力停留时间代理 |
| `WELL_LEVEL_CHANGE` | 清水池水位变化 |
| `FILT. NTU` | 滤后水浊度 |
| `R/W FLOW` | 原水流量 |
| `T/W FLOW` | 出厂水流量 |

辅助物理特征为：

$$
HRT_t=
\frac{L_t}{Q_{out,t}+\varepsilon}
$$

$$
\Delta L_t=L_t-L_{t-1}
$$

应分别训练：

- `RW as Qin`；
- `TW as Qin`。

不建议提前把入流项和出流项合并为净负荷，因为分开输入可以让 GRU 学习两者不同的滞后时间和非线性权重。

## 4. 序列样本构造

采用过去 24 小时作为输入窗口，即过去 12 个观测点：

$$
X_t=
[x_{t-11},x_{t-10},\ldots,x_t]
$$

未来六步增量作为输出：

$$
Y_t=
[\Delta C_{t+1},\Delta C_{t+2},\ldots,\Delta C_{t+6}]
$$

张量形状为：

```text
输入：[batch_size, 12, 8]
输出：[batch_size, 6]
```

GRU 可以通过历史序列学习：

- 入流浊度进入清水池后的传递滞后；
- 清水池水位和流量变化产生的调蓄作用；
- 入流项和出流项的累积影响；
- 质量守恒代理公式未覆盖的非线性关系。

## 5. GRU 模型结构

推荐采用直接多步 GRU：

```text
过去24小时公式分项序列
          ↓
单层GRU
hidden_size = 64
          ↓
最后时刻隐藏状态
          ↓
Dropout
          ↓
全连接层
          ↓
未来六步NTU增量
          ↓
current_NTU + predicted_delta
          ↓
未来2--12小时NTU
```

基本结构可以表示为：

$$
h_t=GRU(x_t,h_{t-1})
$$

$$
\widehat{\Delta\mathbf C}_t
=
W_oh_t+b_o
$$

$$
\hat C_{t+h}
=
C_t+\widehat{\Delta C}_{t+h},
\qquad h=1,\ldots,6
$$

## 6. 物理跳连残差 GRU

还可以进一步构造物理约束更强的模型：

$$
\widehat{\Delta C}_{t+h}
=
a_h I_t-b_h O_t+r_{t,h}
$$

其中：

- $a_h$：第 $h$ 个预测步的入流项权重；
- $b_h$：第 $h$ 个预测步的出流项权重；
- $r_{t,h}$：GRU 学习的非线性残差。

最终预测为：

$$
\hat C_{t+h}
=
C_t+a_hI_t-b_hO_t+r_{t,h}
$$

该结构的解释是：

- 物理分支保留质量守恒的基本方向；
- GRU 残差分支修正非理想混合、水力滞后和代理变量误差；
- $a_h$、$b_h$ 反映入流和出流作用随预测时间的变化。

## 7. 训练策略

### 7.1 数据划分

按照时间顺序划分：

```text
训练集：前70%
验证集：中间15%
测试集：最后15%
```

不能随机划分，以免产生时间信息泄漏。

### 7.2 标准化

- 特征标准化参数仅由训练集计算；
- 目标增量单独标准化；
- 验证集、测试集使用训练集的标准化参数。

### 7.3 损失函数

推荐使用 Huber 损失：

$$
L_{\delta}(e)=
\begin{cases}
\frac{1}{2}e^2,& |e|\leq\delta,\\
\delta(|e|-\frac{1}{2}\delta),& |e|>\delta.
\end{cases}
$$

相比 MSE，Huber Loss 不容易被少量极端 NTU 峰值主导。

### 7.4 推荐参数

| 参数 | 推荐值 |
|---|---:|
| `lookback` | 12 |
| `horizon` | 6 |
| `hidden_size` | 64 |
| `num_layers` | 1 |
| `dropout` | 0--0.2 |
| `batch_size` | 32 |
| `learning_rate` | 0.001 |
| `epochs` | 100 |
| `early_stopping_patience` | 10--15 |

## 8. 直接多步与滚动预测

推荐直接多步输出：

$$
X_t
\longrightarrow
[\hat C_{t+1},\ldots,\hat C_{t+6}]
$$

不优先使用滚动预测，原因包括：

1. 滚动预测会累积前一步误差；
2. 预测未来时刻时，未来流量和水位可能无法提前获得；
3. 直接多步模型可以分别学习不同 horizon 的影响；
4. 更适合输出未来 2--12 小时完整预测结果。

## 9. 对照实验

建议至少比较以下模型：

| 模型 | 说明 |
|---|---|
| 持久性模型 | 所有未来值均等于当前 `NTU` |
| 普通 GRU | 原始变量序列 |
| 公式分项 GRU | 使用入流项、出流项等分项特征 |
| 物理残差 GRU | 物理跳连与 GRU 残差结合 |
| RF/XGBoost | `p3_inv2` 中的单步树模型参考 |

公式分项 GRU 和物理残差 GRU 均应分别测试：

```text
RW as Qin
TW as Qin
```

## 10. 评价指标

每个预测步长分别报告：

- MAE；
- RMSE；
- $R^2$；
- `NTU` Pearson 相关系数；
- `NTU` Spearman 相关系数；
- $\Delta NTU$ Pearson 相关系数；
- $\Delta NTU$ Spearman 相关系数；
- 变化方向准确率。

变化方向准确率定义为：

$$
DA_h=
\frac{1}{N}
\sum_{i=1}^{N}
I\left[
\operatorname{sign}(\Delta C_{i,h})
=
\operatorname{sign}(\widehat{\Delta C}_{i,h})
\right]
$$

应重点观察：

- 相对持久性模型的 RMSE 改善；
- 对未来 NTU 增量的相关性；
- 对上升、下降方向的识别能力。

若绝对 NTU 的相关性很高，但增量相关性接近 0，说明模型主要依赖当前 NTU 的惯性，没有真正学习流量和物料衡算信息。

## 11. 推荐的最终方案

第三题推荐采用：

> 以过去 24 小时的当前出厂水浊度、物料衡算入流项、出流项、水力停留时间代理、清水池水位变化及流量状态组成多变量时间序列，输入 GRU 模型，直接预测未来 2、4、6、8、10 和 12 小时相对于当前出厂水浊度的变化量，再与当前 NTU 相加得到未来多步预测结果。

核心流程为：

```text
原始水质与运行数据
        ↓
质量守恒公式逐项拆分
        ↓
构造过去24小时特征序列
        ↓
直接多步残差GRU
        ↓
预测未来六步NTU增量
        ↓
与当前NTU相加
        ↓
获得未来2--12小时出厂水NTU
```

该方案同时具备：

- 质量守恒方面的物理解释；
- GRU 对时间滞后和累积效应的建模能力；
- 未来六步直接预测能力；
- 对模型是否真正捕捉动态变化的检验能力。
