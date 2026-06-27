# TPMS 及其变种的参数化建模与性能仿真方法

## 1. 研究对象：TPMS 及其变种

TPMS（Triply Periodic Minimal Surface，三周期极小曲面）是一类在三个空间方向上周期重复、平均曲率接近零的连续曲面。本文关注的不是单一曲面展示，而是将 TPMS 作为可调控多孔结构单胞，用于建立“几何参数-空间结构-等效性能”之间的映射关系。

在结构库中，每一个样本由一个 TPMS 拓扑类型和一组几何参数共同确定。基础拓扑包括 Schwarz P、Schwarz D（Diamond）、Gyroid、Neovius、I-WP 等；变种则来自同一拓扑下的参数扰动、等值面偏移、厚度变化、尺度变化、各向异性拉伸、组合/裁剪和不同周期阵列方式。换句话说，TPMS 类型决定孔道连通方式和骨架拓扑，参数决定具体孔隙率、相对密度、比表面积、各向异性和局部曲率分布。

常见 TPMS 可用隐式函数近似描述。例如 Schwarz P 可写为：

$$
f_P(x,y,z)=\cos \frac{2\pi x}{a}
+\cos \frac{2\pi y}{a}
+\cos \frac{2\pi z}{a}
$$

其零水平面为 \(f_P(x,y,z)=C\)。其中 \(a\) 为单胞尺寸，\(C\) 为等值面偏移参数。类似地，Schwarz D、Gyroid、Neovius 和 I-WP 可由不同隐式函数或经审计的 Weierstrass/STL 几何给出。对于正式仿真，几何来源应绑定到已审计的 TPMS 结构或原始参数表，而不是只根据名称临时生成不可追溯的曲面。

## 2. 从参数到结构：建模流程

TPMS 建模的核心是把一行参数表转换为一个确定的三维结构。一个样本可记为：

$$
\mathcal{G}_i = \mathcal{M}(T_i, \theta_i)
$$

其中 \(T_i\) 表示 TPMS 拓扑类型，\(\theta_i\) 表示几何参数向量。参数通常包括：

- 拓扑类型：Schwarz P、Schwarz D、Gyroid、Neovius、I-WP 及其变种编号；
- 周期参数：单胞尺寸 \(a_x,a_y,a_z\) 或周期长度；
- 等值面参数：水平集偏移 \(C\)；
- 厚度参数：薄壁结构中的半厚度或阈值 \(t\)；
- 变形参数：不同方向的缩放、剪切、旋转或形态插值参数；
- 离散参数：体素分辨率、阈值、是否周期边界。

若采用隐式建模，先在周期单胞 \(\Omega\) 中计算水平集函数 \(f(x,y,z)\)，再通过阈值构造实体区域。对于薄壁 TPMS，固体域可写为：

$$
\Omega_s=\{\mathbf{x}\in\Omega:\ |f(\mathbf{x})-C|\le t\}
$$

流体/孔隙域为：

$$
\Omega_f=\Omega\setminus\Omega_s
$$

若采用片体或 shellular 结构，\(t\) 控制壁厚；若采用 solid-network 或 skeletal 结构，则参数控制骨架半径或实体填充区域。实际计算中，为了与求解器兼容，连续结构会被离散为规则体素矩阵：

$$
\chi_s(\mathbf{x})=
\begin{cases}
1, & \mathbf{x}\in\Omega_s \\
0, & \mathbf{x}\in\Omega_f
\end{cases}
$$

本项目的力学与热学标签计算均采用同一体素结构作为输入。也就是说，同一个 `model_id + T` 文件名对应同一个 TPMS 参数行、同一个二值体素矩阵、同一个拓扑结构；区别只在于后续求解器和材料参数不同。

几何建模之后，可从体素结构直接得到基本几何描述量：

$$
\rho^\*=\frac{V_s}{V},\quad
\phi=1-\rho^\*,\quad
a_s=\frac{A_{sf}}{V}
$$

其中 \(\rho^\*\) 为相对密度，\(\phi\) 为孔隙率，\(a_s\) 为比表面积。这些指标是连接结构与性能的重要中间变量。

## 3. TPMS 变种如何进入数据集

所谓“TPMS 变种”不是另起一套无关结构，而是在同一建模函数或同一几何来源上改变参数，使其形成可比较的结构族。可以分为四类：

1. 拓扑变种：不同基础 TPMS 类型，例如 P、D、G、Neovius、I-WP。
2. 水平集变种：改变 \(C\) 或阈值，使孔隙率和连通通道偏移。
3. 厚度变种：改变 \(t\)，调节相对密度、壁厚和承载能力。
4. 几何变形变种：对单胞进行拉伸、压缩、方向缩放、形态插值或裁剪阵列，形成各向异性结构。

因此，每个样本的本质是：

$$
\text{参数} \rightarrow \text{TPMS 类型/变种} \rightarrow \text{体素结构}
$$

同一结构后续可以进入力学均匀化或热学均匀化，从而得到一一对应的力学标签和热学标签。

## 4. 从结构到性能：力学仿真

力学标签的目标是计算 TPMS 单胞的等效弹性张量。输入为二值体素结构，固体相赋予基体材料参数 \(E_s,\nu_s\)，孔隙相视为空相或极弱相。通过周期边界条件下的均匀化计算，得到宏观平均应力-应变关系：

$$
\langle \sigma \rangle = \mathbf{C}^{\mathrm{eff}} \langle \varepsilon \rangle
$$

其中 \(\mathbf{C}^{\mathrm{eff}}\) 是等效弹性刚度张量。数值上，对单胞施加若干独立的宏观应变工况，求解局部位移场和应力场，再体积平均得到张量分量。对三维各向异性材料，完整弹性张量在 Voigt 记号下可表示为 \(6\times6\) 矩阵：

$$
\mathbf{C}^{\mathrm{eff}}=
\begin{bmatrix}
C_{11}&C_{12}&C_{13}&C_{14}&C_{15}&C_{16}\\
C_{21}&C_{22}&C_{23}&C_{24}&C_{25}&C_{26}\\
C_{31}&C_{32}&C_{33}&C_{34}&C_{35}&C_{36}\\
C_{41}&C_{42}&C_{43}&C_{44}&C_{45}&C_{46}\\
C_{51}&C_{52}&C_{53}&C_{54}&C_{55}&C_{56}\\
C_{61}&C_{62}&C_{63}&C_{64}&C_{65}&C_{66}
\end{bmatrix}
$$

在当前服务器流程中，力学标签由 `homo3d` 求解。流程为：

$$
\text{TPMS 参数行}
\rightarrow
\text{体素 bin}
\rightarrow
\text{homo3d}
\rightarrow
\text{ET 标签文件}
$$

输出文件 `ET/model_id_Tk.txt` 代表该结构的等效弹性性能。后续可进一步从 \(\mathbf{C}^{\mathrm{eff}}\) 派生等效杨氏模量、体积模量、剪切模量、各向异性指标和稳定性筛选指标。

## 5. 从结构到性能：热学仿真

热学标签的目标是计算同一 TPMS 单胞的等效热导率张量。输入结构必须与力学完全一致，即同一个 `model_id + T` 对应同一个体素矩阵。区别在于，热学计算给两相赋予不同热导率：

$$
k(\mathbf{x})=
\begin{cases}
k_s, & \mathbf{x}\in\Omega_s \\
k_v, & \mathbf{x}\in\Omega_f
\end{cases}
$$

其中 \(k_s\) 为固体相热导率，\(k_v\) 为空相或低导热相热导率。为了保证数值对比明显并避免过于接近导致病态识别，本项目初始设置为：

$$
k_s=10,\quad k_v=0.01
$$

在周期均匀化框架下，宏观热流和宏观温度梯度满足：

$$
\langle \mathbf{q}\rangle
=
-\mathbf{K}^{\mathrm{eff}}\langle \nabla T\rangle
$$

其中 \(\mathbf{K}^{\mathrm{eff}}\) 为 \(3\times3\) 等效热导率张量：

$$
\mathbf{K}^{\mathrm{eff}}=
\begin{bmatrix}
K_{xx}&K_{xy}&K_{xz}\\
K_{yx}&K_{yy}&K_{yz}\\
K_{zx}&K_{zy}&K_{zz}
\end{bmatrix}
$$

数值上，分别施加 \(x,y,z\) 三个方向的单位宏观温度梯度，求解周期单胞内的温度扰动场和热流场，再体积平均得到张量各分量。服务器上使用 `openTM` 执行该计算：

$$
\text{TPMS 参数行}
\rightarrow
\text{同名体素 bin}
\rightarrow
\text{openTM}
\rightarrow
\text{TT 标签文件}
$$

输出文件 `TT/model_id_Tk.txt` 与 `ET/model_id_Tk.txt` 一一对应，前者为热学张量，后者为力学张量。

## 6. 参数-结构-性能数据闭环

最终数据集的逻辑关系为：

$$
\theta_i
\rightarrow
\chi_i(\mathbf{x})
\rightarrow
\left(
\mathbf{C}^{\mathrm{eff}}_i,
\mathbf{K}^{\mathrm{eff}}_i,
\rho^\*_i,
\phi_i,
a_{s,i}
\right)
$$

其中：

- \(\theta_i\)：原始参数表中的 TPMS 类型和几何参数；
- \(\chi_i(\mathbf{x})\)：由参数生成的二值体素结构；
- \(\mathbf{C}^{\mathrm{eff}}_i\)：力学等效弹性张量；
- \(\mathbf{K}^{\mathrm{eff}}_i\)：热学等效热导率张量；
- \(\rho^\*_i,\phi_i,a_{s,i}\)：几何描述指标。

这样可以建立两个方向的关系：

1. 参数到结构：分析哪些参数控制孔隙率、相对密度、连通性和各向异性。
2. 结构到性能：分析同一结构的刚度、导热、各向异性和多目标权衡。

因此，本文不是孤立地计算一个 TPMS 曲面，而是构建一个可追溯的结构-性能数据集。每个样本都有明确的原始参数、可重建的三维结构、对应的力学张量标签和热学张量标签，为后续代理模型训练、结构筛选和多目标优化提供基础。

## 7. 可直接用于汇报的简述

本研究将 TPMS 及其变种视为由参数控制的周期多孔单胞。首先从原始参数表读取 TPMS 拓扑类型、等值面偏移、厚度和变形参数，生成对应的二值体素结构；然后对同一体素结构分别进行力学和热学均匀化计算。力学部分通过 `homo3d` 得到等效弹性张量，热学部分通过 `openTM` 得到等效热导率张量。两类标签采用完全一致的 `model_id + T` 命名，因此可以保证同一结构下“参数-结构-力学性能-热学性能”的一一对应关系。
