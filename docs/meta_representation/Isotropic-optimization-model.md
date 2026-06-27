# 拓扑优化模型与条件自回归逆向设计

本文整理用于微结构数据生成的拓扑优化模型，并给出条件自回归逆向设计器的简洁建模方式。内容分为三部分：各向同性优化、各向异性扩展、目标性质条件生成。

---

## 1. 各向同性优化模型

拓扑优化在给定设计域内寻找最优材料分布，常用于周期微结构生成。本文采用 SIMP（Solid Isotropic Material with Penalization）材料插值，并在体积分数、边界和各向同性约束下优化等效弹性性能。

### 1.1 基本优化问题

以最大化体积模量 $K$ 为例，优化问题写为：

$$
\begin{aligned}
\min_{\boldsymbol{\rho}}\quad
& f(\mathbf C(\boldsymbol{\rho}))
=-\frac{1}{3}\left(C_{1111}+2C_{1122}\right) \\
\text{s.t.}\quad
& \mathbf K(\boldsymbol{\rho})\mathbf U=\mathbf F,\\
& \frac{\sum_{e=1}^{M}v_e\rho_e}{|\Omega|}\le V,\\
& \phi(b_1-b_2)<\epsilon_1,\\
& (Z-1)^2<\epsilon_2,\\
& 0\le\rho_e\le1,\qquad e=1,\dots,M.
\end{aligned}
$$

其中：

- $\boldsymbol{\rho}$：SIMP 设计变量，即单元材料密度；
- $\rho_e$：第 $e$ 个单元的密度，$\rho_e=0$ 表示空材料，$\rho_e=1$ 表示实体材料；
- $\mathbf K(\boldsymbol{\rho})$：整体刚度矩阵；
- $\mathbf U$：位移场；
- $\mathbf F$：外载荷；
- $v_e$：第 $e$ 个单元体积；
- $|\Omega|$：设计域总体积；
- $V$：给定体积分数上限；
- $b_1,b_2$：材料边界与预设边界；
- $Z$：各向同性指标。

体积分数约束控制材料用量，边界约束使材料边界接近预设边界，各向同性约束使 $Z$ 接近 1。基体材料通常设置为 $E_0=1\times10^6$、$\nu=0.3$。

### 1.2 不同性能目标

为覆盖更丰富的性能分布，数据生成中使用多个目标函数。除最大化体积模量 $K$ 外，还包括最大化剪切模量 $G$ 和杨氏模量 $E$。

最大化剪切模量：

$$
f(\mathbf C(\boldsymbol{\rho}))=-C_{1212}.
$$

最大化杨氏模量：

$$
f(\mathbf C(\boldsymbol{\rho}))
=-\frac{(C_{1111}-C_{1122})(C_{1111}+2C_{1122})}
{C_{1111}+C_{1122}}.
$$

这些目标共享相同的平衡方程、体积分数、边界和密度约束。

### 1.3 指定泊松比

为扩大泊松比分布，可在最大化体积模量的基础上加入目标泊松比约束：

$$
\begin{aligned}
\min_{\boldsymbol{\rho}}\quad
& -\frac{1}{3}\left(C_{1111}+2C_{1122}\right) \\
\text{s.t.}\quad
& \mathbf K(\boldsymbol{\rho})\mathbf U=\mathbf F,\\
& \frac{\sum_{e=1}^{M}v_e\rho_e}{|\Omega|}\le V,\\
& \phi(b_1-b_2)<\epsilon,\\
& (Z-1)^2<\epsilon,\\
& \left(
\frac{C_{1122}}{C_{1111}+C_{1122}}-v_0
\right)^2<\epsilon,\\
& 0\le\rho_e\le1,\qquad e=1,\dots,M.
\end{aligned}
$$

其中 $v_0$ 为目标泊松比。对于各向同性微结构，泊松比理论上限为 0.5。接近该上限时，普通约束优化较困难，通常需要专门的五模结构目标。

### 1.4 五模结构目标

为生成接近泊松比上限的五模结构，可使用如下目标：

$$
f(\mathbf C(\boldsymbol{\rho}))
=-K+
\frac{G-0.8G_0}
{K\left(0.5\frac{G_0}{K_0}\right)+0.05}.
$$

其中 $K$ 和 $G$ 分别为当前结构的体积模量与剪切模量，$K_0$ 和 $G_0$ 为初始随机结构的对应值。分母中的常数 $0.05$ 用于避免奇异性。

---

## 2. 各向异性优化模型

各向异性微结构可产生负泊松比、方向增强模量等特殊性质。为扩充数据集，可去掉各向同性约束，仅保留平衡、体积分数、边界和密度约束。

以最大化体积模量为例：

$$
\begin{aligned}
\min_{\boldsymbol{\rho}}\quad
& -\frac{1}{3}\left(C_{1111}+2C_{1122}\right) \\
\text{s.t.}\quad
& \mathbf K(\boldsymbol{\rho})\mathbf U=\mathbf F,\\
& \frac{\sum_{e=1}^{M}v_e\rho_e}{|\Omega|}\le V,\\
& \phi(b_1-b_2)<\epsilon,\\
& 0\le\rho_e\le1,\qquad e=1,\dots,M.
\end{aligned}
$$

最大化剪切模量、杨氏模量和五模结构可沿用第 1 节的目标函数，只是不再施加 $(Z-1)^2<\epsilon$。

指定泊松比的各向异性模型为：

$$
\begin{aligned}
\min_{\boldsymbol{\rho}}\quad
& -\frac{1}{3}\left(C_{1111}+2C_{1122}\right) \\
\text{s.t.}\quad
& \mathbf K(\boldsymbol{\rho})\mathbf U=\mathbf F,\\
& \frac{\sum_{e=1}^{M}v_e\rho_e}{|\Omega|}\le V,\\
& \phi(b_1-b_2)<\epsilon,\\
& \left(
\frac{C_{1122}}{C_{1111}+C_{1122}}-v_0
\right)^2<\epsilon,\\
& 0\le\rho_e\le1,\qquad e=1,\dots,M.
\end{aligned}
$$

---

## 3. 条件自回归逆向设计

逆向设计任务可建模为：给定目标力学性能，逐步生成微结构表示。对于二维体素结构，模型直接预测材料状态序列；对于三维 B 样条表示，模型可预测控制系数序列。

### 3.1 输入与输出

目标性能向量记为：

$$
\mathbf p^\ast=
\left[
K^\ast,\ G^\ast,\ E^\ast,\ \nu^\ast,\ V^\ast,\ Z^\ast
\right].
$$

二维二值微结构展平成序列：

$$
\mathbf x=[x_1,x_2,\dots,x_N],\qquad x_i\in\{0,1\}.
$$

其中 $N=H\times W$，$x_i=1$ 表示实体材料，$x_i=0$ 表示空隙。

三维 B 样条逆向设计也可写成同一形式，只需将输出序列替换为控制系数：

$$
C\rightarrow(c_1,c_2,\dots,c_T),\qquad T=4096.
$$

### 3.2 自回归分解

模型学习条件分布：

$$
p_\theta(\mathbf x\mid \mathbf p^\ast)
=\prod_{i=1}^{N}
p_\theta(x_i\mid \mathbf x_{<i},\mathbf p^\ast).
$$

其中 $\mathbf x_{<i}=[x_1,\dots,x_{i-1}]$。该分解使模型在预测当前位置时同时依赖目标性能和已生成结构前缀。

### 3.3 网络结构

网络由三部分组成：

```text
Property Encoder
  -> Conditional AR Transformer
  -> Structure Prediction Head
```

性能编码器将目标性能映射为条件嵌入：

$$
\mathbf c=\mathcal E_p(\mathbf p^\ast).
$$

结构序列经过 token embedding 和位置编码：

$$
\mathbf e_i=\operatorname{Embed}(x_i)+\mathbf e_i^{\mathrm{pos}}.
$$

带 causal mask 的 Transformer 根据结构前缀与条件嵌入计算隐藏状态：

$$
\mathbf h_i=\mathcal T_\theta(\mathbf x_{<i},\mathbf c).
$$

条件可通过三种方式注入：

- 条件 token：把 $\mathbf c$ 放在序列开头；
- cross-attention：把 $\mathbf c$ 作为 condition memory；
- FiLM：用 $\gamma(\mathbf c)$ 和 $\beta(\mathbf c)$ 调制中间特征。

预测头输出当前位置为实体材料的概率：

$$
\hat x_i
=\sigma(\mathbf w^\top \mathbf h_i+b)
=p_\theta(x_i=1\mid \mathbf x_{<i},\mathbf p^\ast).
$$

### 3.4 训练目标

训练采用 teacher forcing：输入真实前缀，预测完整序列。二值结构使用二元交叉熵：

$$
\mathcal L_{\mathrm{AR}}
=-\sum_{i=1}^{N}
\left[
x_i\log\hat x_i+(1-x_i)\log(1-\hat x_i)
\right].
$$

为提高目标性能一致性，可加入冻结的性能预测器 $\mathcal P_\psi$：

$$
\hat{\mathbf p}=\mathcal P_\psi(\hat{\mathbf x}),
$$

并定义：

$$
\mathcal L_{\mathrm{prop}}
=\left\|\hat{\mathbf p}-\mathbf p^\ast\right\|_2^2.
$$

体积分数约束：

$$
\mathcal L_{\mathrm{vol}}
=\left(
\frac{1}{N}\sum_{i=1}^{N}\hat x_i-V^\ast
\right)^2.
$$

二值化正则：

$$
\mathcal L_{\mathrm{bin}}
=\frac{1}{N}\sum_{i=1}^{N}\hat x_i(1-\hat x_i).
$$

总损失为：

$$
\mathcal L
=\lambda_{\mathrm{AR}}\mathcal L_{\mathrm{AR}}
+\lambda_{\mathrm{prop}}\mathcal L_{\mathrm{prop}}
+\lambda_{\mathrm{vol}}\mathcal L_{\mathrm{vol}}
+\lambda_{\mathrm{bin}}\mathcal L_{\mathrm{bin}}.
$$

### 3.5 推理流程

推理时仅输入目标性能 $\mathbf p^\ast$，模型从起始 token 开始逐步采样：

$$
\hat x_i\sim p_\theta(x_i\mid \hat{\mathbf x}_{<i},\mathbf p^\ast),
\qquad i=1,\dots,N.
$$

得到候选结构后，通过有限元均匀化验证真实性能：

$$
\mathbf p(\hat{\mathbf x})
=\left[
K(\hat{\mathbf x}),\ G(\hat{\mathbf x}),\ E(\hat{\mathbf x}),\
\nu(\hat{\mathbf x}),\ V(\hat{\mathbf x}),\ Z(\hat{\mathbf x})
\right].
$$

性能误差可写为：

$$
\mathcal E_p=\left\|\mathbf p(\hat{\mathbf x})-\mathbf p^\ast\right\|.
$$

---

## 4. 简洁表述

本文将微结构逆向设计建模为条件自回归生成任务。给定目标性能 $\mathbf p^\ast$，模型按预定义空间顺序逐步预测结构状态：

$$
p_\theta(\mathbf x\mid \mathbf p^\ast)
=\prod_{i=1}^{N}
p_\theta(x_i\mid \mathbf x_{<i},\mathbf p^\ast).
$$

目标性能先由编码器映射为条件嵌入 $\mathbf c=\mathcal E_p(\mathbf p^\ast)$，再由带因果掩码的 Transformer 结合结构前缀计算隐藏表示：

$$
\mathbf h_i=\mathcal T_\theta(\mathbf x_{<i},\mathbf c).
$$

结构预测头输出实体概率：

$$
\hat x_i=\sigma(\mathbf w^\top \mathbf h_i+b).
$$

模型通过自回归重建损失、性能一致性损失、体积分数约束和二值化正则联合训练，使生成结构在可生成性的基础上尽量接近目标力学性能。
