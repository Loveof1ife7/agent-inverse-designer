# TPMS Inverse Designer

本文整理 TPMS 逆向设计器的核心思路：先无条件学习有效 TPMS Fourier 系数的结构流形，再在目标性质条件下生成候选系数，最后通过确定性 TPMS 解码、性能预测和高精度仿真筛选得到最终结构。

---

## 1. 核心目标

给定目标性质 $y$，生成一个 TPMS 单胞结构：

```math
S=(c,\rho).
```

其中：

- $c$：TPMS 隐式场的 Fourier 系数；
- $\rho$：由 $c$ 解码得到的二值密度结构；
- $y$：目标条件，例如体积分数、体积模量、剪切模量、杨氏模量、泊松比或完整弹性张量。

目标是学习条件分布：

```math
p_\theta(c\mid y).
```

使生成的 Fourier 系数既落在有效 TPMS 结构流形上，又能解码出匹配目标性能的结构。

---

## 2. 表示方式

TPMS 结构由周期隐式场定义。用有限阶 Fourier 系数表示该隐式场：

```math
\phi_c(x)=\sum_{m\in\mathcal M}(a_m\cos(2\pi k_m^\top x)+b_m\sin(2\pi k_m^\top x)),\quad x\in[0,1]^3.
```

其中：

- $\mathcal M$：保留的频率集合；
- $k_m$：第 $m$ 个频率向量；
- $a_m,b_m$：对应 Fourier 系数；
- $c=\{a_m,b_m\}_{m\in\mathcal M}$：完整系数向量。

隐式场通过阈值化或平滑投影得到密度：

```math
\rho(x)=H_{\eta}(\phi_c(x)).
```

阈值 $\eta$ 可根据目标体积分数 $V_f^*$ 自动调整，使：

```math
\frac{1}{|\Omega|}\int_{\Omega}\rho(x)\,dx\approx V_f^*.
```

这个表示的好处是：周期性由 Fourier 基函数天然保证，模型只需要学习低维系数空间，而不必直接生成高分辨率体素。

---

## 3. 总体架构

TPMS inverse designer 分为两个训练阶段和一个推理阶段：

```text
Stage 1: Unconditional pretraining
  -> learn the valid TPMS Fourier coefficient manifold

Stage 2: Conditional inverse-design training
  -> learn target-property-conditioned coefficient generation

Inference:
  -> sample candidates
  -> decode TPMS density
  -> rank by surrogate
  -> verify by homogenization
```

整体数据流为：

```text
target property y
  -> condition encoder
  -> conditional coefficient generator
  -> Fourier coefficients c
  -> TPMS implicit decoder
  -> density rho
  -> property surrogate / homogenization
  -> final verified structure
```

核心思想是把“生成有效几何”和“匹配目标性质”拆开：

- 无条件预训练负责学习 TPMS 系数空间中哪些结构像真实有效样本；
- 条件训练在这个流形上引导生成结果靠近目标性能；
- 解码和均匀化保持为确定性物理流程，避免网络直接幻想最终性能。

---

## 4. Stage 1: 无条件预训练

无条件预训练只看 Fourier 系数 $c$，不输入目标性质。目标是学习有效 TPMS 系数的先验分布：

```math
p_\theta(c).
```

训练数据为：

```math
\mathcal D_c=\{c^{(n)}\}_{n=1}^{N}.
```

根据具体生成模型，训练目标可以写成不同形式：

### 4.1 Autoregressive prior

若将系数量化或按固定顺序展开为序列：

```math
c\rightarrow(c_1,c_2,\dots,c_T).
```

则学习：

```math
p_\theta(c)=\prod_{t=1}^{T}p_\theta(c_t\mid c_{<t}).
```

训练损失为负对数似然：

```math
\mathcal L_{\mathrm{prior}}=-\sum_{t=1}^{T}\log p_\theta(c_t\mid c_{<t}).
```

### 4.2 Diffusion or flow prior

若使用 diffusion 或 flow matching，则模型学习从噪声到系数分布的生成路径。目标仍然是得到一个可采样的先验：

```math
z\sim\mathcal N(0,I),\quad c=G_\theta(z).
```

这一阶段不追求目标性质控制，只要求生成结果落在合理 TPMS 系数流形上。

---

## 5. Stage 2: 条件逆向设计训练

条件训练在预训练先验的基础上加入目标性质 $y$。目标分布为：

```math
p_\theta(c\mid y).
```

目标条件可写为：

```math
y=(V_f^*,p^*,t).
```

其中：

- $V_f^*$：目标体积分数；
- $p^*$：目标力学性质；
- $t$：可选的 TPMS family、symmetry 或任务类型标签。

首先用 condition encoder 得到条件嵌入：

```math
e_y=\mathrm{Enc}_y(y).
```

然后将 $e_y$ 注入生成器。常见方式包括：

- 条件 token：把 $e_y$ 作为序列开头 token；
- cross-attention：把 $e_y$ 作为 generator 的 condition memory；
- FiLM / AdaLN：用 $e_y$ 调制中间特征；
- classifier-free guidance：训练时随机丢弃条件，推理时增强条件控制。

条件生成目标为：

```math
\mathcal L_{\mathrm{cond}}=-\log p_\theta(c\mid y).
```

为了提升目标性质匹配能力，可加入冻结的性能预测器：

```math
\hat p=f_{\mathrm{prop}}(c,V_f^*,t).
```

性质一致性损失为：

```math
\mathcal L_{\mathrm{prop}}=||\hat p-p^*||_2^2.
```

同时加入系数正则，使条件生成结果不要偏离预训练流形：

```math
\mathcal L_{\mathrm{prior}}^{\mathrm{reg}}=-\log p_{\theta_0}(c).
```

其中 $p_{\theta_0}$ 是 Stage 1 得到的冻结先验。

总损失可写为：

```math
\mathcal L=\mathcal L_{\mathrm{cond}}+\lambda_{\mathrm{prop}}\mathcal L_{\mathrm{prop}}+\lambda_{\mathrm{prior}}\mathcal L_{\mathrm{prior}}^{\mathrm{reg}}.
```

---

## 6. 确定性 TPMS 解码

生成器只输出 Fourier 系数 $c$。最终结构由确定性解码器得到：

```text
Fourier coefficients c
  -> implicit field phi_c(x)
  -> threshold eta from target volume fraction
  -> binary density rho
  -> homogenized properties
```

解码过程为：

```math
c\rightarrow\phi_c(x)\rightarrow\rho(x)\rightarrow C^{\mathrm{hom}}.
```

这里 $C^{\mathrm{hom}}$ 是周期均匀化得到的等效弹性张量。模型不直接输出体素或弹性张量，而是输出可复现的结构参数。

---

## 7. 推理流程

给定目标：

```text
target volume fraction: Vf*
target properties: p*
optional family / symmetry / task type: t
```

推理步骤为：

```text
1. Encode target:
       e_y = Enc_y(Vf*, p*, t)

2. Sample M coefficient candidates:
       c^(1), ..., c^(M) ~ p_theta(c | y)

3. Decode each candidate:
       phi_c(x) -> rho(x)

4. Predict properties by surrogate:
       p_hat = f_prop(c, Vf*, t)

5. Rank candidates:
       score = property error + validity penalty + volume penalty

6. Select top candidates:
       keep top K structures

7. Run high-resolution reconstruction and homogenization:
       rho_hd -> C_hom -> verified properties

8. Output final verified TPMS structures.
```

候选排序可写为：

```math
\mathrm{score}(c)=||\hat p(c)-p^*||_2^2+\alpha|\hat V_f(c)-V_f^*|+\beta\,\mathrm{invalid}(c).
```

---

## 8. 输出格式

最终输出建议包含：

```text
generated_coeffs: c
tpms_family: optional family label
target_volume_fraction: Vf*
target_properties: p*
decoded_density: rho
predicted_properties: p_hat
homogenized_properties: p_hom
validity_flags
score
```

其中：

- `generated_coeffs` 是模型直接生成的 Fourier 系数；
- `decoded_density` 是确定性 TPMS 解码后的体素结构；
- `predicted_properties` 是 surrogate 快速预测结果；
- `homogenized_properties` 是高分辨率均匀化后的真实性能。

---

## 9. 总结

TPMS inverse designer 的核心不是直接生成高维体素，而是生成低维 Fourier 系数：

```text
target properties
  -> conditional generator
  -> Fourier coefficients
  -> TPMS implicit field
  -> binary density
  -> surrogate screening
  -> homogenization verification
```

两阶段训练让任务更稳定：无条件预训练先学会“什么是有效 TPMS 系数”，条件训练再学习“怎样根据目标性质移动到合适的系数区域”。这样可以同时保持结构有效性、周期性和目标性质可控性。
