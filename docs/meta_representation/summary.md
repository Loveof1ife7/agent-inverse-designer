# Meta Representation Summary

可以把四类结构表示统一成同一个三段式：

$$
\boxed{
\text{结构参数 } \theta
\xrightarrow{\ \mathcal{D}\ }
\text{结构数据 } X
\xrightarrow{\ \mathcal{S}\ }
\text{仿真性能 } y
}
$$

其中：

- $\theta$：可调的结构参数；
- $\mathcal{D}$：数据生成器，把参数变成几何、体素或杆系；
- $X$：仿真输入结构数据；
- $\mathcal{S}$：FEM 或均匀化仿真器；
- $y$：性能标签。

如果要优化，就是：

$$
\boxed{
\max_{\theta} J(y)
=
\max_{\theta} J\left(\mathcal{S}(\mathcal{D}(\theta))\right)
}
$$

也就是通过修改结构参数 $\theta$，让仿真性能 $y$ 变好。

---

## 1. Truss 结构

### 结构参数

Truss 表示中，单个具体结构的完整参数是：

$$
\boxed{
\theta_{\mathrm{truss}} = (G, u_G, \mathcal{E})
}
$$

其中：

- $G$：三维晶体群；
- $u_G$：群约束后的自由点位参数；
- $\mathcal{E}$：代表点之间的杆件集合。

原始点位参数是：

$$
p \in [0,1]^{27}
$$

群约束给出：

$$
A_G p = b_G
$$

所以真正独立的几何变量是：

$$
p = B_G u_G + c_G
$$

因此，Truss 的结构变量可以理解为：

$$
\boxed{
u_G \text{ 控制节点位置，}\quad
\mathcal{E} \text{ 控制杆件拓扑。}
}
$$

注意：$\theta_{\mathrm{truss}}$ 是单个结构的完整表示；当前 closed-loop controller 实际更常优化的是生成分布参数，例如 `group`、`max_bars`、`rho_target` 和采样范围。DATA-GEN 再从这些搜索参数中采样具体的 $(G,u_G,\mathcal{E})$。

### 数据生成

$$
(G,u_G,\mathcal{E})
\rightarrow
p
\rightarrow
19\text{ 个代表点}
\rightarrow
\text{群扩展}
\rightarrow
\text{单胞杆系}
\rightarrow
4\times4\times4\text{ 扩胞}
$$

最终生成：

$$
\boxed{
X_{\mathrm{truss}}
=
(\texttt{node\_data},\texttt{element\_conn})
}
$$

其中：

- `node_data`：节点坐标；
- `element_conn`：杆件连接关系。

### 仿真性能

Truss 使用 Abaqus/Explicit 做大变形压缩仿真：

$$
(\texttt{node\_data},\texttt{element\_conn})
\rightarrow
\text{梁单元模型}
\rightarrow
\sigma-\varepsilon
\rightarrow
y_{\mathrm{truss}}
$$

性能标签为：

$$
\boxed{
y_{\mathrm{truss}}
=
(E,\sigma_{\text{peak}},\sigma_{fp},W_v,
\sigma_{pl},\mathrm{CFE},
\mathrm{drop\_ratio},\delta,\varepsilon_{cd})
}
$$

完整链条：

$$
\boxed{
(G,u_G,\mathcal{E})
\rightarrow
(\texttt{node\_data},\texttt{element\_conn})
\rightarrow
\sigma-\varepsilon
\rightarrow
9\text{ 个压缩性能指标}
}
$$

---

## 2. B-spline 隐式表示

### 结构参数

B-spline 表示中，结构参数可写成：

$$
\boxed{
\theta_{\mathrm{bspline}} = (G, C_{\mathrm{ctrl}}, V_f)
}
$$

其中：

- $G$：空间群或平面群；
- $C_{\mathrm{ctrl}}$：B 样条控制系数；
- $V_f$：目标体积分数。

三维中：

$$
C_{\mathrm{ctrl}} \in \mathbb{R}^{16\times16\times16}
$$

它定义隐式场：

$$
\Psi(u,v,w)
=
\sum_{i,j,k}
(C_{\mathrm{ctrl}})_{ijk}
N_i(u)N_j(v)N_k(w)
$$

再通过群轨道平均得到对称场：

$$
\phi(x)
=
\frac{1}{|G|}
\sum_{g\in G}
\Psi(gx)
$$

所以 B-spline 的调优变量主要是：

$$
\boxed{
C_{\mathrm{ctrl}}
}
$$

如果跨空间群筛选，则 $G$ 也是调优变量。若采用 standardized embedding v1，lattice embedding 可以视为固定常量；`objective` 更像生成来源或优化目标的 provenance，不一定是结构重建所需的最小参数。

### 数据生成

$$
(G,C_{\mathrm{ctrl}},V_f)
\rightarrow
\Psi
\rightarrow
\phi
\rightarrow
\rho
\rightarrow
\text{二值体素结构}
$$

其中：

$$
\rho(x)=H_{\beta,\eta}(\phi(x))
$$

最终生成：

$$
\boxed{
X_{\mathrm{bspline}}
=
\texttt{density\_final}
}
$$

也就是一个三维体素密度场，例如：

$$
\texttt{density\_final}\in\{0,1\}^{256\times256\times256}
$$

### 仿真性能

B-spline 结构进入周期性均匀化：

$$
\texttt{density\_final}
\rightarrow
\text{周期均匀化}
\rightarrow
\mathbf{C}^{\mathrm{eff}}
\rightarrow
y_{\mathrm{bspline}}
$$

性能标签包括：

$$
\boxed{
y_{\mathrm{bspline}}
=
(
\mathbf{C}_{\mathrm{voigt}},
\mathbf{C}_{\mathrm{kelvin}},
K,G,
E_x,E_y,E_z,
\nu,
A_{\mathrm{universal}}
)
}
$$

完整链条：

$$
\boxed{
(G,C_{\mathrm{ctrl}},V_f)
\rightarrow
\texttt{density\_final}
\rightarrow
\mathbf{C}^{\mathrm{eff}}
\rightarrow
\text{等效弹性性能}
}
$$

---

## 3. 密度单元场表示

### 结构参数

密度单元场，也就是 SIMP 拓扑优化表示，其结构参数直接是单元密度：

$$
\boxed{
\theta_{\rho}=\boldsymbol{\rho}
}
$$

其中：

$$
\boldsymbol{\rho}
=
(\rho_1,\rho_2,\dots,\rho_M)
$$

并且：

$$
0\leq \rho_e\leq1
$$

$\rho_e$ 表示第 $e$ 个有限元单元的材料密度。

它和 B-spline 的区别是：

$$
\boxed{
\text{B-spline 调控制系数 } C_{\mathrm{ctrl}},
\quad
\text{密度单元场直接调 } \rho_e。
}
$$

B-spline 可以理解为低维参数化密度场；SIMP density field 则是直接以每个单元密度作为设计变量的高维表示。

### 数据生成

密度场本身就是结构数据：

$$
\boldsymbol{\rho}
\rightarrow
\text{密度场}
\rightarrow
\text{SIMP 材料插值}
$$

SIMP 插值可写成：

$$
E_e(\rho_e)
=
E_{\min}
+
\rho_e^p(E_0-E_{\min})
$$

最终生成：

$$
\boxed{
X_{\rho}
=
\boldsymbol{\rho}
\text{ 或二值化后的密度结构}
}
$$

如果后处理清洗或二值化，则：

$$
\boldsymbol{\rho}
\rightarrow
\chi_s(x)
\in\{0,1\}
$$

### 仿真性能

密度单元场进入有限元均匀化：

$$
\boldsymbol{\rho}
\rightarrow
\mathbf{K}(\boldsymbol{\rho})
\rightarrow
\mathbf{K}(\boldsymbol{\rho})\mathbf{U}=\mathbf{F}
\rightarrow
\mathbf{C}^{\mathrm{eff}}(\boldsymbol{\rho})
\rightarrow
y_{\rho}
$$

性能包括：

$$
\boxed{
y_{\rho}
=
(K,G,E,\nu,\mathbf{C}^{\mathrm{eff}},
\text{各向同性指标},
\text{稳定性指标})
}
$$

完整链条：

$$
\boxed{
\boldsymbol{\rho}
\rightarrow
\text{SIMP 密度结构}
\rightarrow
\mathbf{C}^{\mathrm{eff}}
\rightarrow
\text{等效弹性性能}
}
$$

---

## 4. TPMS 表示

### 结构参数

TPMS 表示中，结构参数是：

$$
\boxed{
\theta_{\mathrm{TPMS}}=(T,\eta)
}
$$

其中：

- $T$：TPMS 类型；
- $\eta$：几何参数向量。

例如：

$$
T\in
\{
\text{Schwarz P},
\text{Schwarz D},
\text{Gyroid},
\text{Neovius},
\text{I-WP}
\}
$$

几何参数可以写成：

$$
\eta
=
(a_x,a_y,a_z,C,t,s_x,s_y,s_z,\ldots)
$$

其中：

- $a_x,a_y,a_z$：周期尺寸；
- $C$：等值面偏移；
- $t$：壁厚或阈值；
- $s_x,s_y,s_z$：各向异性缩放；
- 其他参数：旋转、剪切、裁剪、形态插值等。

### 数据生成

TPMS 通过隐式函数生成结构：

$$
(T,\eta)
\rightarrow
f_T(x,y,z;\eta)
\rightarrow
\Omega_s
\rightarrow
\chi_s(x)
$$

薄壁 TPMS 可写成：

$$
\Omega_s
=
\{
x\in\Omega:
|f_T(x;\eta)-C|\leq t
\}
$$

二值体素结构为：

$$
\chi_s(x)
=
\begin{cases}
1,&x\in\Omega_s\\
0,&x\notin\Omega_s
\end{cases}
$$

最终生成：

$$
\boxed{
X_{\mathrm{TPMS}}
=
\chi_s(x)
}
$$

也就是 TPMS 的二值体素结构。

### 仿真性能

同一个 TPMS 体素结构可以进入力学和热学两个仿真链路。

力学：

$$
\chi_s(x)
\rightarrow
\texttt{homo3d}
\rightarrow
\mathbf{C}^{\mathrm{eff}}
$$

热学：

$$
\chi_s(x)
\rightarrow
\texttt{openTM}
\rightarrow
\mathbf{K}^{\mathrm{eff}}
$$

所以性能为：

$$
\boxed{
y_{\mathrm{TPMS}}
=
(
\mathbf{C}^{\mathrm{eff}},
\mathbf{K}^{\mathrm{eff}},
\rho^*,
\phi,
a_s
)
}
$$

其中：

- $\mathbf{C}^{\mathrm{eff}}$：等效弹性张量；
- $\mathbf{K}^{\mathrm{eff}}$：等效热导率张量；
- $\rho^*$：相对密度；
- $\phi$：孔隙率；
- $a_s$：比表面积。

完整链条：

$$
\boxed{
(T,\eta)
\rightarrow
\chi_s(x)
\rightarrow
(\mathbf{C}^{\mathrm{eff}},\mathbf{K}^{\mathrm{eff}})
\rightarrow
\text{力学/热学性能}
}
$$

---

## 5. 四种表示的统一总表

| 类型 | 结构参数 $\theta$ | 数据生成 $X=\mathcal{D}(\theta)$ | 仿真性能 $y=\mathcal{S}(X)$ |
| --- | --- | --- | --- |
| Truss 结构 | $(G,u_G,\mathcal{E})$ | `node_data + element_conn` | Abaqus 压缩曲线与 9 个大变形指标 |
| B-spline | $(G,C_{\mathrm{ctrl}},V_f)$ | `density_final` 体素密度 | 均匀化弹性张量与弹性标量 |
| 密度单元场 | $\boldsymbol{\rho}$ | SIMP 密度场或二值密度场 | 均匀化弹性张量与 $(K,G,E,\nu)$ |
| TPMS | $(T,\eta)$ | TPMS 二值体素结构 | 力学张量、热学张量、几何指标 |

---

## 6. 统一数学形式

四类都可以写成：

$$
\boxed{
\theta
\rightarrow
X
\rightarrow
y
}
$$

具体为：

$$
\boxed{
\theta_{\mathrm{truss}}
=(G,u_G,\mathcal{E})
\rightarrow
X_{\mathrm{truss}}
=(\texttt{node\_data},\texttt{element\_conn})
\rightarrow
y_{\mathrm{truss}}
}
$$

$$
\boxed{
\theta_{\mathrm{bspline}}
=(G,C_{\mathrm{ctrl}},V_f)
\rightarrow
X_{\mathrm{bspline}}
=\texttt{density\_final}
\rightarrow
y_{\mathrm{bspline}}
}
$$

$$
\boxed{
\theta_{\rho}
=\boldsymbol{\rho}
\rightarrow
X_{\rho}
=\text{density field}
\rightarrow
y_{\rho}
}
$$

$$
\boxed{
\theta_{\mathrm{TPMS}}
=(T,\eta)
\rightarrow
X_{\mathrm{TPMS}}
=\chi_s(x)
\rightarrow
y_{\mathrm{TPMS}}
}
$$

---

## 7. Closed-loop Discovery 中的对应关系

在 closed-loop discovery 中，系统做的是：

$$
\boxed{
y^*
\rightarrow
\theta_{\mathrm{search}}
\rightarrow
\mathcal{D}(\theta_{\mathrm{search}})
\rightarrow
X
\rightarrow
\mathcal{S}(X)
\rightarrow
y
\rightarrow
\text{feedback}
\rightarrow
\theta_{\mathrm{search}}'
}
$$

其中：

- $y^*$：目标性能；
- $\theta_{\mathrm{search}}$：AgentExplorer / InverseDesigner 可直接提议的搜索参数；
- $\mathcal{D}$：DATA-GEN 或其他结构生成器；
- $X$：可仿真的结构数据；
- $\mathcal{S}$：FEM、均匀化或 proxy evaluator；
- $y$：评价得到的性能；
- `FeedbackSignal`：比较 $y$ 与 $y^*$，抽取下一轮控制信号。

对当前 Truss 系统尤其要区分：

$$
\boxed{
\theta_{\mathrm{search}}
\neq
\theta_{\mathrm{truss}}
}
$$

当前 controller 通常提议的是：

$$
\theta_{\mathrm{search}}
=
(\texttt{group},\texttt{max\_bars},\texttt{rho\_target},
\texttt{parameter\_ranges},\texttt{constraints},\ldots)
$$

而 DATA-GEN 落地后产生具体结构：

$$
\theta_{\mathrm{truss}}
=(G,u_G,\mathcal{E})
$$

也就是说：

```text
AgentExplorer proposes target schedule S_n = [T_1, ..., T_k].
InverseDesigner samples explicit structures for scheduled targets.
FEMEvaluator evaluates the generated structures.
KnowledgeBase stores schedule -> structure -> property evidence.
FeedbackSignal compares evaluated properties against the final target.
```

Note: `DatagenFEMEvaluator` remains the offline data factory for cold-start
pretraining. It should not be the online exploration engine in the new
closed-loop design.

---

## 8. 最终统一理解

四种表示的差异主要在于**结构参数是什么**：

$$
\boxed{
\text{Truss：调群约束下的节点位置和杆连接}
}
$$

$$
\boxed{
\text{B-spline：调隐式场的控制系数}
}
$$

$$
\boxed{
\text{密度单元场：直接调每个单元的材料密度}
}
$$

$$
\boxed{
\text{TPMS：调拓扑类型和少量几何参数}
}
$$

但它们的统一逻辑完全一致：

$$
\boxed{
\text{选择或修改结构参数}
\rightarrow
\text{生成可仿真的结构数据}
\rightarrow
\text{通过 FEM 或均匀化得到性能}
\rightarrow
\text{根据性能反过来优化结构参数}
}
$$
