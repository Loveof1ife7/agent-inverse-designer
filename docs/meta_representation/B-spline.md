# 三维晶体群约束 B 样条超材料表示

本文说明项目中超材料单胞的统一表示：如何用周期 B 样条控制系数定义结构，如何施加空间群约束，如何保存二维/三维数据，以及如何通过周期均匀化得到力学标签。

项目目标是在二维平面群和三维空间群约束下批量生成周期超材料单胞，并形成可用于机器学习、逆向设计和材料筛选的结构-性能数据集。

---

## 1. 数学表示

### 1.1 参数域与晶胞嵌入

三维单胞定义在归一化参数域：

$$
\Omega_p=[0,1]^3,\qquad x=(u,v,w).
$$

物理晶胞由三条晶格基向量张成，晶胞参数为：

$$;f
(a,b,c,\alpha,\beta,\gamma).
$$

当前三维主数据集采用 **standardized embedding v1**，不使用自然晶胞轴长，而使用统一归一化嵌入：

| 晶系 | 晶胞参数 |
|---|---|
| triclinic / monoclinic / orthorhombic / tetragonal / cubic | $a=b=c=1,\ \alpha=\beta=\gamma=90^\circ$ |
| trigonal / hexagonal | $a=b=c=1,\ \alpha=\beta=90^\circ,\ \gamma=120^\circ$ |

这样可以统一二维/三维解释方式，避免人为轴长比例带来的差异，并让不同空间群下的数据处在可比较的几何尺度中。非正交或非等长 metric 可作为扩展实验，但不是主数据集默认表示。

### 1.2 B 样条控制系数

结构不直接以体素为设计变量，而由较低维的 B 样条控制系数表示连续隐式场。三维控制系数为：

$$
C=\{C_{ijk}\},\qquad i=1,\dots,n_u,\ j=1,\dots,n_v,\ k=1,\dots,n_w.
$$

当前配置为：

$$
n_u=n_v=n_w=16,
$$

因此每个三维结构由 $16^3=4096$ 个控制系数描述。给定三次均匀 B 样条基函数 $N_{i,3}$，未加群约束的隐式场为：

$$
\Psi(u,v,w)=\sum_i\sum_j\sum_k C_{ijk}N_{i,3}(u)N_{j,3}(v)N_{k,3}(w).
$$

二维版本为：

$$
\Psi(u,v)=\sum_i\sum_j C_{ij}N_{i,3}(u)N_{j,3}(v).
$$

也就是说，二维使用控制矩阵，三维使用控制张量。

### 1.3 周期性

为了让单胞可无缝平铺，控制系数按周期索引使用：

$$
C_{ijk}=C_{i\bmod n_u,\ j\bmod n_v,\ k\bmod n_w}.
$$

二维情形同理：

$$
C_{ij}=C_{i\bmod n_u,\ j\bmod n_v}.
$$

三次循环 B 样条保证隐式场在周期边界处的函数值、一阶导数和二阶导数连续。最终结构不是离散噪声，而是由跨单胞边界连续的隐式函数定义。

### 1.4 空间群约束

给定目标空间群 $G$，每个群元素 $g$ 作用于参数域坐标。循环 B 样条已经保证平移周期性，因此只需对有限群操作代表元做轨道平均：

$$
\phi(x)=\frac{1}{|G|}\sum_{g\in G}\Psi(gx).
$$

该操作使任意点的场值由其空间群轨道共同决定，因此结构在表示层面天然满足指定空间群对称性，而不是生成后再做对称性检查。

### 1.5 从隐式场到二值结构

对称隐式场 $\phi$ 通过平滑 Heaviside 投影转为物理密度：

$$
\rho=H_{\beta,\eta}(\phi).
$$

其中 $\eta$ 为阈值，$\beta$ 控制投影陡峭程度。优化早期使用较小 $\beta$ 保持可微，后期逐步增大 $\beta$，使密度接近二值。最终保存：

$$
\rho(x)\in\{0,1\},
$$

其中 0 表示孔洞，1 表示实体材料。

### 1.6 三维样本定义

一个三维样本可概括为：

$$
\text{sample}=\big(G,\ C,\ (a,b,c,\alpha,\beta,\gamma),\ V_f,\ \text{objective}\big).
$$

其中：

- $G$：空间群编号；
- $C$：$16\times16\times16$ B 样条控制系数张量；
- $(a,b,c,\alpha,\beta,\gamma)$：standardized embedding v1 下的晶胞参数；
- $V_f$：目标体积分数；
- `objective`：生成结构时使用的正向或逆向目标。

给定这些量后，连续隐式场、二值密度结构和后续均匀化标签均被确定。

---

## 2. 生成策略

### 2.1 低维控制空间

设计变量是 B 样条控制系数，而不是 $256^3$ 个体素。当前三维配置为：

| 项目 | 设置 |
|---|---|
| 控制点 | $16\times16\times16$ |
| 优化网格 | $32\times32\times32$ |
| 高分辨率重构 | $256\times256\times256$ |
| B 样条次数 | 3 |
| 体积分数范围 | 约 $0.10$ 到 $0.70$ |

流程采用“低分辨率优化、高分辨率标注”：先在 $32^3$ 网格上优化控制系数，再在 $256^3$ 网格上重构结构并重新计算性能标签。

### 2.2 正向生成

正向生成直接采样或扰动控制系数，再重构结构并计算性能。主要用于扩大样本数量和覆盖局部几何变体。

- 随机正向：在控制系数空间直接采样；
- 种子微扰：对优秀逆向样本加小扰动：

$$
C'=\operatorname{clip}(C+\epsilon,0,1).
$$

随后重新重构和标注，以保留优质主拓扑并增加局部多样性。

### 2.3 逆向生成

逆向生成给定目标体积分数和目标力学性质，通过优化控制系数寻找满足目标的结构。当前目标包括：

| 目标 | 含义 |
|---|---|
| `target_K` | 追踪指定体积模量 |
| `Max_Bulk` | 最大化体积模量 |
| `Max_Shear` | 最大化剪切模量 |
| `Auxetic` | 倾向负泊松比或负耦合刚度 |
| `Unimode_EasyShear` | 倾向难压缩、易剪切 |
| `HardShear_EasyBulk` | 倾向难剪切、易压缩 |

多目标设计的重点不是单纯增加数量，而是覆盖更丰富的弹性张量分布。

### 2.4 样本筛选

保存到正式数据集前，样本需满足：

- 密度足够接近二值；
- 最终体积分数在允许范围内；
- 周期连通性和局部连通性通过；
- 均匀化计算得到有限、有效的弹性张量；
- Kelvin 表示下的弹性矩阵特征值可用于正定性检查；
- 正式数据以 H5 为主，不保存体积巨大的 VTK 文件。

失败样本不进入正式数据集。

---

## 3. 数据结构

### 3.1 三维 `.h5` 样本

一个 `.h5` 文件对应一个三维结构样本，同时保存结构、参数、标签和来源信息。顶层结构为：

```text
/identity
/lattice
/bspline
/density
/elasticity
/properties
/diagnostics
/provenance
```

| 组 | 主要字段 | 说明 |
|---|---|---|
| `/identity` | `sample_id`, `spacegroup_number`, `spacegroup_symbol`, `crystal_system` | 样本身份与空间群信息 |
| `/lattice` | `params`, `vectors`, `cell_volume` | 晶胞参数与基向量 |
| `/bspline` | `coeffs_optimized`, `control_grid_shape`, `degree` | 低维 B 样条表示 |
| `/density` | `density_final`, `density_raw`, `grid_shape`, `actual_volume_fraction`, `target_volume_fraction` | 体素密度结构 |
| `/elasticity` | `C_voigt`, `C_kelvin`, `kelvin_eigenvalues` | 周期均匀化弹性张量 |
| `/properties` | `K_bulk_hd`, `G_shear_hd`, `Ex`, `Ey`, `Ez`, `nu_*`, `A_universal` 等 | 标量力学性能 |
| `/diagnostics`, `/provenance` | 目标函数参数、误差、版本、来源 | 复现与排查信息 |

其中 `/lattice/params` 必须符合 standardized embedding v1：

```text
非三方/六方: [1,1,1,90,90,90]
三方/六方:   [1,1,1,90,90,120]
```

`density_final` 为最终二值结构，0 表示孔洞，1 表示实体材料。

Voigt 顺序为：

```text
xx, yy, zz, yz, xz, xy
```

Kelvin 表示用于弹性张量正定性检查：

$$
C_{\mathrm{Kelvin}}=D C_{\mathrm{Voigt}}D,\qquad
D=\operatorname{diag}(1,1,1,\sqrt2,\sqrt2,\sqrt2).
$$

### 3.2 二维 `.mat` 样本

一个 `.mat` 文件对应一个二维结构样本，核心字段为：

| 字段 | 含义 |
|---|---|
| `Q` | 高清结构重新均匀化后的 $3\times3$ 等效刚度矩阵 |
| `xPhys_HD_Final` | 高清二值或清洗后的二维密度结构 |
| `coffi_total` | 最终 B 样条控制系数 |
| `real_volfrac` | 最终真实体积分数 |
| `Target_K_Input` | 输入目标体积模量 |
| `M_nd` | 非离散度/灰度指标 |
| `gray_status` | 灰度质量状态 |
| `nelx,nely` | 优化阶段网格尺寸 |
| `hd_nelx,hd_nely` | 高清重构网格尺寸 |

二维派生性能由柔度矩阵 $S=Q^{-1}$ 计算：

$$
E_x=\frac{1}{S_{11}},\qquad
E_y=\frac{1}{S_{22}},\qquad
G_{xy}=\frac{1}{S_{33}},
$$

$$
\nu_{xy}=-S_{21}E_x,\qquad
\nu_{yx}=-S_{12}E_y,
$$

$$
K=\frac{1}{\sum S_{1:2,1:2}}.
$$

### 3.3 二维与三维对应关系

| 含义 | 二维项目 | 三维项目 |
|---|---|---|
| 群约束 | 17 个平面群 | 230 个空间群 |
| 参数表示 | B 样条控制矩阵 | B 样条控制张量 |
| 最终结构 | 二维密度图 | 三维体素密度 |
| 主结构文件 | `.mat` | `.h5` |
| 弹性标签 | $3\times3$ 矩阵 $Q$ | $6\times6$ 矩阵 $C$ |
| 主要标量 | $E_x,E_y,G_{xy},K,\nu$ | $E_x,E_y,E_z,G,K,\nu,A$ |

三维项目本质上是二维隐式张量积 B 样条方法在空间群约束和三维均匀化上的扩展。

---

## 4. 周期均匀化

### 4.1 二维均匀化

二维单胞被离散为有限元网格，材料插值采用 SIMP：

$$
k_e(\rho_e)=\left(E_{\min}+\rho_e^p(E_0-E_{\min})\right)k_0.
$$

需施加三个独立宏观单位应变工况：

```text
1. x 方向拉伸
2. y 方向拉伸
3. xy 剪切
```

每个工况使用周期性边界条件求解微观位移场，再通过能量均匀化计算等效刚度：

$$
Q_{ij}=\frac{1}{|Y|}\sum_e (u_e^{(i)})^T k_e u_e^{(j)}.
$$

### 4.2 三维均匀化

三维输入为最终二值体素密度场 $\rho(x,y,z)$。默认归一化材料参数为：

```text
solid Young's modulus E0 = 1
solid Poisson ratio nu = 0.3
void stiffness Emin 很小
SIMP penalty p = 3
```

三维均匀化覆盖六个独立宏观应变分量：

```text
xx, yy, zz, yz, xz, xy
```

求解后得到 $6\times6$ 等效弹性矩阵 $C_{\mathrm{Voigt}}$，再派生 Kelvin 表示、特征值和标量力学性质。

---

## 5. 条件自回归逆向设计

逆向设计器学习从目标性质到 B 样条控制系数的条件分布。给定目标：

$$
y^\ast=(G,\ V_f^\ast,\ p^\ast),
$$

其中 $G$ 为空间群，$V_f^\ast$ 为目标体积分数，$p^\ast$ 为目标力学性质。模型输出：

$$
C\in[0,1]^{16\times16\times16}.
$$

将控制张量按固定顺序展开为长度 $T=4096$ 的序列：

$$
C\rightarrow(c_1,c_2,\dots,c_T).
$$

自回归模型学习：

$$
p_\theta(C\mid y^\ast)
=\prod_{t=1}^{T}p_\theta(c_t\mid c_{<t},y^\ast).
$$

生成的控制系数再经过确定性解码：

$$
\rho
=H_{\beta,\eta(V_f^\ast)}
\left[
\mathcal P_G\left(\mathcal B_{\mathrm{per}}(C)\right)
\right],
$$

其中 $\mathcal B_{\mathrm{per}}$ 表示周期 B 样条重构，$\mathcal P_G$ 表示空间群轨道平均。周期性和空间群对称性由解析算子保证，而不是由神经网络隐式学习。

推理流程为：

```text
target properties
  -> conditional AR generator
  -> B-spline control tensor C
  -> periodic B-spline reconstruction
  -> space-group orbit averaging
  -> density projection
  -> candidate structure
  -> surrogate screening
  -> high-resolution homogenization
  -> final property-verified structure
```

输出字段建议包括：

```text
generated_coeffs: C, shape = [16,16,16]
decoded_density: rho, shape = [256,256,256]
spacegroup: G
target_volume_fraction: Vf*
target_properties: p*
predicted_properties: p_hat
homogenized_properties: p_hom
validity_flags
```

---

## 6. 总结

本项目的核心链路为：

```text
空间群/平面群
  -> 周期性隐式张量积 B 样条参数化
  -> 群轨道平均得到对称隐式场
  -> Heaviside 投影得到二值单胞结构
  -> 周期性有限元均匀化
  -> 等效弹性张量与标量性能标签
  -> `.mat` / `.h5` 单样本数据
```

简言之：用低维、连续、带群约束的 B 样条控制系数表示复杂周期超材料结构，再用周期均匀化将结构转化为可学习、可筛选的弹性性能数据。
