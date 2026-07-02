# CMA-ES 反演材料参数 — 仿真实验对齐流水线

**状态(2026-06-16):**
- 🟢 server 上 4 个单 case 反演中:14245 P222 (g187) / 14246 Pbar4m2 ✅g300 (best=4.1e-4) / 14247 I422 (g284) / 14248 P422 (g293)
- 🟢 **新材料 LC_20260615 matA**:JC + Damage 6 参数拟合 RMS=0.1911 MPa (1.30%),详见 §8,**沿用 JC 框架上线 server**
- 通用脚本 **`D:\组会\3Dtruss大变形\build\dynamic_B31_JC_generic.py`** 严禁动
- JC 模型仍是当前 server 实际跑的(过去试错全删,只保留当前可用骨干)

---

## 1. 路径

### 服务器(`/public/home/qingfang/cma_invert/`)
```
master/
  cma_jc_invert_server.py          # 反演主控,加 --case + --x0-14195-best + pickle Resume
  dynamic_B31_JC_generic.py        # 通用脚本副本(等于本地版,不动)
inputs/exp/                        # 4 case xlsx 实验数据
work/g{gen:03d}_c{cand:02d}_{case}/ # 每候选每 case 临时目录(自动清 odb)
results_<case>/                    # 4 个独立目录:history.json + cma_state.pkl + std.out/err
slurm/job_cma_<case>.slurm         # 4 份 sbatch
```

### 本地(只读引用)
- 通用脚本:`D:\组会\3Dtruss大变形\build\dynamic_B31_JC_generic.py`
- 4 case lattice txt:`C:\Users\admin\Desktop\3Dtruss\outputs\<群>\crystal_4x4x4\<id>.txt`
- 4 case STL:`D:\组会\3Dtruss大变形\实验打印\<群>\<id>.stl`
- 4 case 实验 xlsx:`D:\组会\3Dtruss大变形\实验与仿真对比\<case>_<日期>.xlsx`
- 新材料 LC_20260615 原 zip + 解压:`D:\Download\20260615LC.zip` / `D:\组会\3Dtruss大变形\材料拟合\LC_20260615\20260615LC\`

---

## 2. 单位制(写死,不准搞错)

abaqus mm-tonne-s-N:长度 mm / 质量 t / 时间 s / 力 N → σ = MPa,E/A/B 都是 MPa,ν/n 无量纲,密度 ρ_TPU = 1.11e-9 t/mm³。

---

## 3. 当前架构(单 case 独立反演,JOBID 14245-14248)

### 3.1 配置

| 项 | 值 |
|---|---|
| 维度 | 6 (E, ν, A, B, n, µ) |
| 起步 θ | `--x0-14195-best`:14195 联合 g96 best (E=487.7, ν=0.39, A=36.3, B=186, n=5.16, µ=0.261) |
| σ0 | 0.25 |
| popsize | 32 |
| N_GEN | 300 |
| LB | (50, 0.01, 1, 10, 0.1, 0.01) |
| UB | (5000, 0.49, 200, 2000, 10, 0.99) |
| W_ANCHOR | 1.0 |
| ANCHOR_WINDOW | 2 grid (±0.0207 应变) |
| EXTREMA_ORDER | 5(argrelextrema 局部极值滤波) |
| EPS_GRID | linspace(0, 0.30, 30) |
| CPUS_PER_CASE | 1 |

### 3.2 节点分配(满载 128 cpu)

| sbatch | case | 节点 | cpu |
|---|---|---|---|
| 14245 cma_P222_38429 | P222_38429 | cnode1 | 32 (32 cand × 1 case × 1 cpu) |
| 14246 cma_Pbar4m2_4799 | Pbar4m2_4799 | cnode1 | 32 |
| 14247 cma_I422_7871 | I422_7871 | cnode2 | 32 |
| 14248 cma_P422_15689 | P422_15689 | cnode2 | 32 |

`--overlap` 允许同节点 2 个 sbatch 共享 64 cpu,无排队。

### 3.3 sbatch 模板(`slurm/job_cma_<case>.slurm`,4 份)

```bash
#!/bin/bash
#SBATCH -J cma_<case>
#SBATCH -p CPU
#SBATCH -N 1
#SBATCH -w <cnode1 或 cnode2>
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=100g
#SBATCH --time 72:00:00
#SBATCH -o /public/home/qingfang/cma_invert/results_<case>/std.out.%j
#SBATCH -e /public/home/qingfang/cma_invert/results_<case>/std.err.%j

source /public/home/qingfang/.bashrc
conda activate abaqus
cd /public/home/qingfang/cma_invert
/public/home/qingfang/.conda/envs/abaqus/bin/python /public/home/qingfang/cma_invert/master/cma_jc_invert_server.py --case <case> --x0-14195-best
```

### 3.4 master 内 srun 调度(关键不能错)

```python
# cma_jc_invert_server.py line ~213-219
['srun', '-n', '1', '-N', '1', '--overlap', f'-c{CPUS_PER_CASE}',
 '--mem=0', f'--nodelist={NODES[cand_idx % len(NODES)]}',
 '/public/home/qingfang/.conda/envs/abaqus/bin/python', script_dst]
```

要点:
- `--overlap`:允许 step 共享节点资源(同节点多 sbatch 不会排队)
- `--mem=0`:srun step 不占内存配额(由 sbatch 总配额 100g 分)
- `--nodelist=NODES[cand_idx % len(NODES)]`:round-robin 均分到节点。单节点 sbatch (-w cnodeX) 时 NODES=['cnodeX'],所有 srun 都落在指定节点
- **python 必须用绝对路径**:`/public/home/qingfang/.conda/envs/abaqus/bin/python`,**不能用裸 `python`**(SLURM 节点 PATH 空,会 ERR exec)。14153 因此爆炸 24h 全 ERR

### 3.5 单代时间 + 总时长

| case | 单代时间 | 300 代 |
|---|---|---|
| Pbar4m2(最简,已完成) | ~255s | ~21h |
| I422 / P422 | ~530s | ~44h |
| P222(最复杂) | ~820s | ~68h(72h 上限刚够)|

---

## 4. Loss 公式(当前 JC 反演)

```
per_case = mse / σ_peak² + W_PEAK·peak_err² + W_EPS·eps_err² + W_ANCHOR·anchor_loss
                                ↑              ↑
                              =0 不激活      =0 不激活
total = mean(per_case 字典)        # 单 case 反演:CASES 只 1 个 → total = per_case
异常 case → per[name] = 1e3 惩罚
```

### 4.1 实验预处理(`load_exp_sigma_eps`,line 103-148)

```
xlsx → F_kgf, D_mm
σ_exp = F_kgf · 9.80665 / A_stl     # STL bbox 校准
ε_exp = D_mm / H_stl
i0 = argmax(σ_exp > SIG_EXP_THRESH=0.05)
ε ← ε[i0:] - ε[i0]                  # ε 和 σ 都减起点,平移到 (0,0)
σ ← σ[i0:] - σ[i0]
排序 + 去重 + np.interp 到 EPS_GRID(30 点)
锚点:argrelextrema(σ_grid, np.greater/less, order=5) + i_peak 强制加入,np.unique 去重
```

### 4.2 仿真裁剪(`cut_and_interp_sim`,line 146-159,**只 ε 平移不 σ 平移**)

```
排序 + 去重 → 找 σ > 1e-6 首点 i_first → i_keep = max(0, i_first-1)
ε_keep = ε_u[i_keep:] - ε_u[i_keep]   # ε 平移到 0
σ_keep = σ_u[i_keep:]                 # σ 不平移(保留物理 σ=0 起点)
sig_grid = np.interp(EPS_GRID, ε_keep, σ_keep, left=0.0, right=σ_keep[-1])
```

### 4.3 锚点窗口(`evaluate_candidate`,line 268-280)

```python
for j in extrema_idx:
    lo = max(0, int(j) - ANCHOR_WINDOW)      # ±2 grid 窗口
    hi = min(N, int(j) + ANCHOR_WINDOW + 1)
    target = sig_exp[int(j)]
    diffs = np.abs(sig_sim[lo:hi] - target)
    i_best = lo + int(diffs.argmin())        # 窗口内找最接近的仿真值
    anchor_errs.append(((sig_sim[i_best] - target) / sig_peak)**2)
loss_anchor = float(np.mean(anchor_errs))
```

允许仿真小相位差(±0.02 应变),形状对了+小平移 → 锚点项归零。

### 4.4 STL bbox 校准(4 case)

| case | A_stl (mm²) | H_stl (mm) |
|---|---|---|
| P222_38429 | 4003.0 | 63.27 |
| Pbar4m2_4799 | 1530.1 | 39.12 |
| I422_7871 | 2685.5 | 51.82 |
| P422_15689 | 2528.1 | 50.28 |

---

## 5. 面积归一化 bug fix(2026-06-13,骨干)

通用脚本 `dynamic_B31_JC_generic.py` line 355-356, 186:
```
area = Lx * Ly                  # lattice 节点 bbox(不含杆半径)
stress = -rf3_eff / AREA        # σ_sim = F / A_txt
```

实验侧 σ_exp = F·g / A_stl(含杆半径,大 7-11%)。两边不一致 → CMA 反演 E/A/B 偏小 7-11%。

**Fix(不改通用脚本,只改反演脚本):**
```python
# run_abaqus_one_case line 234-238
sig_grid = sig_grid * (area / case['A_real_mm2'])
# 等价于 σ_sim = F·(A_txt/A_stl)/A_txt = F/A_stl
```

| case | A_txt | A_stl | A_stl/A_txt |
|---|---|---|---|
| P222_38429 | 3751.5 | 4003.0 | 1.067 |
| Pbar4m2_4799 | 1376.1 | 1530.1 | 1.112 |
| I422_7871 | 2480.2 | 2685.5 | 1.083 |
| P422_15689 | 2329.0 | 2528.1 | 1.085 |

---

## 6. 反演脚本架构(`cma_jc_invert_server.py`)

### 6.1 改动清单

| 改动 | line | 内容 |
|---|---|---|
| `import pickle` | 5 | |
| `N_GEN = 300` | 51 | |
| 面积 fix | 234-238 | `sig_grid *= area / case['A_real_mm2']` |
| `--case <name>` + `results_<case>/` + `--x0-14195-best` | 348-368 | 过滤 CASES + 改 RESULTS + 覆盖 THETA0 |
| pickle Resume 优先 | 395-460 | 详见 §6.2 |
| 每代 dump pickle | 485-491 | 原子写入 `.tmp + os.replace` |

### 6.2 pickle 真断点续算(骨干)

**每代 `pickle.dump(es, ...)` 保存 `cma.CMAEvolutionStrategy` 整个对象,包含:**
- 协方差矩阵 C
- 步长 σ_norm(归一化空间真实当前值,**不是常量 SIGMA0**)
- 演化路径 p_σ, p_c
- 当前均值 m
- 迭代计数 countiter
- ask/tell 队列等所有内部 state

写入路径:`results_<case>/cma_state.pkl`,**原子写入**(`tmp + os.replace`)防中途崩溃损坏。

### 6.3 Resume 三级回退逻辑(line 395-460)

```
1. 优先 pickle:cma_state.pkl 存在 + 维度匹配
   → es = pickle.load(cma_state.pkl)        # 加载完整 state
   → es.opts['maxiter'] = N_GEN              # 上调让 stop() 不再因 maxiter 触发
   → 真断点续算从 countiter 接着跑(σ_norm/C/p_σ/p_c/m 全部接续)

2. 回退 history.json:只有 json 没有 pkl(老 JOBID)
   → cma_x0 = best_theta,σ0 = SIGMA0(常量 0.25 大步长)
   → 热启动重新探索(前 5-10 代会变差再收敛)

3. 全新:都没有 → THETA0 起步
```

### 6.4 main 循环每代(line ~411-491)

```python
gen = start_gen
while not es.stop():
    gen += 1
    X = es.ask()
    thetas = [LB + np.array(x) * (UB - LB) for x in X]
    losses, pers = evaluate_generation(thetas, gen, cases_with_geom, exp_data)
    es.tell(X, losses)
    i_best = int(np.argmin(losses))
    if losses[i_best] < best_loss:
        best_loss = float(losses[i_best])
        best_theta = thetas[i_best].copy()
    history.append(dict(gen=gen, thetas=..., losses=..., per_case=..., best_in_gen=..., best_overall=..., best_theta=...))
    json.dump(history, results_<case>/history.json)
    tmp_pkl = pkl_path + '.tmp'
    with open(tmp_pkl, 'wb') as f: pickle.dump(es, f)
    os.replace(tmp_pkl, pkl_path)              # 原子写入
    print(f'[g{gen:02d}] best_gen=... σ_norm={es.sigma:.4f} ...')
```

### 6.5 通用脚本调用 + csv 输出(关键接口)

```python
# run_abaqus_one_case (line 183-245)
work_dir = WORK_BASE / f'g{gen:03d}_c{cand_idx:02d}_{case_name}'
shutil.copy(GENERIC_SCRIPT, work_dir / 'dynamic_B31_JC_generic.py')
# sed 改副本顶部 10 个用户配置:
#   INPUT_FOLDER = lattice txt 目录
#   TARGET_FILE_INDEX = txt 文件名(无后缀)
#   OUTPUT_FOLDER = work_dir
#   YOUNG_MODULUS / POISSON_RATIO / JC_A / JC_B / JC_N / FRICTION_COEFF / CPUS
subprocess.run(['srun', '-n', '1', '-N', '1', '--overlap', ...])
# 输出 csv:work_dir/<TARGET_FILE_INDEX>/<idx>_PLATE_Z_XYPBC_StackZ1_curve.csv
#   列 0=Strain, 列 1=Disp_cm, 列 2=Force_kN, 列 3=Stress_kNcm2(实际 MPa)
df = pd.read_csv(...)
eps_raw = df.iloc[:, 0].values
sig_raw = df.iloc[:, 3].values                 # 列 3 实际 MPa,列名误标
sig_grid = cut_and_interp_sim(eps_raw, sig_raw)
sig_grid *= area / case['A_real_mm2']          # 面积 fix(§5)
# 清 odb / sta / msg / lck 等大文件,保留 csv
return sig_grid
```

---

## 7. 端到端操作手册(Phase A-F,骨干)

### Phase A:本地改 master 脚本

1. 编辑 `D:\Download\test_hybrid_mesh\cma_jc_invert_server_<日期>.py`
2. 改顶部配置(N_GEN / LB/UB / W_ANCHOR 等)
3. 改 CASES 字典(name / lattice_txt / exp_xlsx / H_stl / A_stl)
4. 不动 evaluate_candidate / cut_and_interp_sim / Resume 段,除非有 bug

### Phase B:上传 server master

```bash
KEY='D:/Download/210.45.73.118_<日期>_rsa.txt'
scp -i $KEY "D:/Download/test_hybrid_mesh/cma_jc_invert_server_<日期>.py" \
    qingfang@210.45.73.118:/public/home/qingfang/cma_invert/master/cma_jc_invert_server.py
ssh -i $KEY qingfang@210.45.73.118 "grep -nE 'N_GEN|import pickle|--case' /public/home/qingfang/cma_invert/master/cma_jc_invert_server.py"
```

### Phase C:备份 history + 清 work(选)

```bash
ssh -i $KEY qingfang@210.45.73.118 "
cd /public/home/qingfang/cma_invert/results_<case>
[ -f history.json ] && mv history.json history_<旧JOBID>.json.bak
[ -f cma_state.pkl ] && mv cma_state.pkl cma_state_<旧JOBID>.pkl.bak"
# work 子目录不动(name 不冲突,自动复用)
```

### Phase D:提交 sbatch(全新 / 续算)

```bash
# 全新(对应 case 的 results_<case>/ 内不存在 history.json 和 pkl):
ssh -i $KEY qingfang@210.45.73.118 "
cd /public/home/qingfang/cma_invert/slurm
sbatch job_cma_<case>.slurm"

# 续算(pkl 已存在):
# 直接 sbatch 同一份脚本即可,自动加载 pkl 真断点续算
```

### Phase E:提交后立即检查(必跑)

```bash
# 等 30s 让 SLURM 启动
sleep 30
ssh -i $KEY qingfang@210.45.73.118 "
squeue -u qingfang -o '%i %j %T %M %L %N'                      # 状态应该是 RUNNING 不是 PENDING
tail -3 /public/home/qingfang/cma_invert/results_<case>/std.err.<JOBID>"
# std.err 不能有 srun 报错 / python ImportError / abaqus 启动失败
```

发现秒停 / std.err 报错 → 立刻 `scancel <JOBID>` 并查根因。

### Phase F:监控 + 拉对比图

详见 §3.4 时间预估 + 下方 §10 监控命令。每 g25 / g50 / g100 拉一次 best 对比图。

---

## 8. 新材料 LC_20260615 matA(JC + Damage 6 参数,关键章节)

### 8.1 原始数据

- **zip**:`D:\Download\20260615LC.zip` → 解压到 `D:\组会\3Dtruss大变形\材料拟合\LC_20260615\20260615LC\`
- 4 个 specimen 文件夹(MTS 拉伸机标准输出),`.dat` 列结构:`Time(s) | Axial Displacement(mm) | Axial Force(N) | Axial Strain(mm/mm)`
- **材料 A**(本节拟合对象):spec608 / spec609,截面 10.4 × 8.7 = **90.48 mm²**,拉到 ε_nomi ≈ 0.48
- **材料 B**(基本单调):spec606 / spec607,截面 8.9 × 3.33 = **29.64 mm²**,拉到 ε_nomi ≈ 0.19
- 处理脚本:`D:\组会\3Dtruss大变形\材料拟合\LC_20260615\process_LC.py`

### 8.2 工程应力 + 平移(0, 0)

```
σ_nomi = F_N / A_mm2  (MPa)
ε_nomi ← ε_nomi - ε_nomi[0]    # 去传感器零漂
σ_nomi ← σ_nomi - σ_nomi[0]    # 减初始预紧 (~ 6-10 N,占峰值 0.05-0.6%)
```

### 8.3 真应力真应变变换(论文 Eq. A1)

```
σ_true = σ_nomi · (1 + ε_nomi)
ε_true = ln(1 + ε_nomi)
```

**matA 真应力曲线特征**(spec608):陡升 → 首峰 → 软化 → 二次升 → 断裂卸载
- ε_true ≈ 0.047 → σ_true ≈ 12.79(首峰)
- ε_true ≈ 0.167 → σ_true ≈ 11.60(平台谷)
- ε_true ≈ 0.39 → σ_true ≈ 14.72(二次峰)
- ε_true > 0.39 → σ_true 暴跌到 0(试样断裂卸载)

**拟合域**:截到 `ε_true ≤ 0.38` 去掉断裂卸载段;原始 22603 点等间距重采样到 **500 点**(`np.interp`)加速优化。

### 8.4 JC + Damage 模型

弹性段 + 塑性段(JC) + 损伤折减,3 段衔接:

```
弹性  e ≤ A/E :  σ = E·e
塑性  e >  A/E :  σ_y(ε_p) = A + B·ε_p^n            (隐式 ε_p = e − σ/E,迭代 80 步收敛 1e-7)
                D(ε_p) = clip((ε_p − ε_pd) / u_f, 0, 1)
                σ = (1 − D)·σ_y
```

形状能力:**1 个首峰 + 1 个谷 + 1 个二次峰**(末段必然降到 0 因 D→1)。**多次振荡 / 永久高位**描述不了。

### 8.5 JC + Damage 6 参数 baseline(matA,当前 baseline,RMS=0.1911 MPa)

**拟合算法**:`scipy.optimize.differential_evolution` 全局搜 + `curve_fit` LM 精修
- forward 用 `@numba.njit(parallel=True, fastmath=True)`(prange 并发 popsize, SIMD pow)
- popsize=80, maxiter=3000(强制跑满,callback 永远 False)
- 5 个 seed (42, 7, 123, 999, 31415) 独立 DE,LM 精修取最佳
- bounds:E∈[100, 2000], A∈[0.1, 50], B∈[0.1, 500], n∈[0.05, 5.0], ε_pd∈[0, 0.4], u_f∈[0.01, 10]
- **wall=105s 全跑完**(numba 多核加速 ~30×)

**当前 baseline 参数**(2026-06-17,5 seed 全收敛同解 = 全局最优强证):

```
E      = 448.471 MPa
A      =  12.4393 MPa
B      = 262.0127
n      =   2.5640
eps_pd =   0.0280
u_f    =   0.6475

RMS = 0.1911 MPa (1.30% of σ_peak=14.718)
所有参数都在 bound 中段,无一顶 UB/LB
```

**对比**:旧 dogbone JC+Damage 卡死 RMS=0.458 (3.37%) 的根因是 n 顶 UB=1.5 不够凹;放开到 5 后 n=2.56 凹形够强,二次升出来,RMS 降到 1.30%(几乎追平当时 Ogden 6 的 1.28%)。

### 8.6 LC_20260615 文件路径(本地)

- 4 specimen csv (7 列 time/disp/force/eps_nomi/sig_nomi/eps_true/sig_true):
  `D:\组会\3Dtruss大变形\材料拟合\LC_20260615\spec60{6,7,8,9}_mat{A,B}_*.csv`
- 真应力对比图(每材料 1 张 + 全 4 条 overlay):
  `D:\组会\3Dtruss大变形\材料拟合\LC_20260615\matA_10.4x8.7.png` / `matB_8.9x3.33.png` / `all4_overlay.png`
- matA JC+Damage 拟合图:`D:\组会\3Dtruss大变形\材料拟合\LC_20260615\matA_JCdamage.png`
- 拟合脚本:`D:\组会\3Dtruss大变形\材料拟合\LC_20260615\_jc_damage_matA.py`

### 8.7 上线 server 路径(已拍板)

**JC + Damage,沿用 `dynamic_B31_JC_generic.py` 框架,通用脚本不动。**

abaqus `*PLASTIC` 卡片原生支持 `*DAMAGE INITIATION` + `*DAMAGE EVOLUTION`,只需在反演脚本里把 (E, ν, A, B, n) 5 参数扩成 (E, ν, A, B, n, ε_pd, u_f, μ) 8 参数(damage 2 个 + 摩擦 µ),并把 sed 替换段补充 damage 卡片。

材料 B(spec606/607)基本单调,首峰后波动 < 2-3%,**直接 JC 4 参数 (E, A, B, n)** 足够。本节不细写,需要时单独拟。

---

## 9. 监控 + 续算命令(骨干)

### 9.1 SSH key 每天 24h 轮换

```bash
KEY='D:/Download/210.45.73.118_<YYYYMMDDHHMMSS>_rsa.txt'
```

### 9.2 查 4 case 进度

```bash
ssh -i $KEY qingfang@210.45.73.118 "squeue -u qingfang -o '%i %j %T %M %L %N'"
for case in P222_38429 Pbar4m2_4799 I422_7871 P422_15689; do
  ssh -i $KEY qingfang@210.45.73.118 \
    "grep -E '^\[g' /public/home/qingfang/cma_invert/results_$case/std.out.* | tail -3"
done
```

### 9.3 找 best θ 所在 g/c

```bash
ssh -i $KEY qingfang@210.45.73.118 "python3 -c '
import json
h=json.load(open(\"/public/home/qingfang/cma_invert/results_<case>/history.json\"))
bt=h[\"best\"][\"theta\"]
for g in h[\"history\"]:
    l=g[\"losses\"]; i=l.index(min(l))
    if all(abs(a-b)<1e-9 for a,b in zip(g[\"thetas\"][i], bt)):
        print(\"g\", g[\"gen\"], \"c\", i, \"loss\", min(l)); break'"
```

### 9.4 拉 best 的仿真 csv

```bash
scp -i $KEY "qingfang@210.45.73.118:/public/home/qingfang/cma_invert/work/g<gen>_c<cand>_<case>/<idx>/<idx>_PLATE_Z_XYPBC_StackZ1_curve.csv" \
    "D:/Download/test_hybrid_mesh/.../sim_<case>.csv"
```

### 9.5 真断点续算

任何时候 `scancel <JOBID>` 后,直接 `sbatch slurm/job_cma_<case>.slurm` → 脚本自动加载 `results_<case>/cma_state.pkl` → 真断点续算从 countiter 接着跑。**无需手动改任何参数**。

---

## 10. 已知风险

- **SLURM srun 跨节点**:必须 `--nodelist=NODES[cand_idx % len(NODES)]` round-robin,否则全堆 cnode1
- **python 必须绝对路径**:`/public/home/qingfang/.conda/envs/abaqus/bin/python`,裸 `python` 会 ERR exec(SLURM 节点 PATH 空,血泪教训 14153)
- **必须 `conda activate abaqus`**:sbatch run.sh 内必须激活,否则缺 numpy/cma
- **abaqus 1 cpu 单仿真**:多线程辅助不挤(已验证 64+64 cpu 满载 OK)
- **work 残留**:旧 JOBID g00x_cxx_<case> 子目录不清(占盘但无害,本次跑用 g<gen>_c<cand>_<case> 不冲突)
- **ssh key 每天 24h 轮换**:监控前确认 key 文件名
- **通用脚本 `dynamic_B31_JC_generic.py` 严禁动**:任何改 abaqus 卡片(加 damage / 改硬化形式 / 改截面 / 改单元等)必须用户授权,且推荐写新文件不动原脚本

---

## 11. JOBID 历史(只标 active / 废)

| JOBID | 状态 |
|---|---|
| 14245-14248 | 🟢 当前有效(单 case + pickle 续算 + N_GEN=300) |
| 14195(g96 best θ) | 仅作 `--x0-14195-best` 起步用,其他全废 |
| 14038 / 14053 / 14117 / 14146 / 14151 / 14166 / 14174 / 14189 | 全废(JC 6D 联合或旧面积归一化) |

---

## 12. Dogbone 材料常数库(Tough 2000 v2 + v1.1)

CMA-ES 反演的 THETA0(E, ν, A, B, n, ε_pd, u_f)来源于 dogbone 单轴拉伸实验的 JC+Damage 拟合。**lattice 实验用哪批 TPU,THETA0 就用对应的 6 参数**,严禁混用。

### Tough 2000 v2(LC_20260615)

- **原始 MTS .dat**:`D:\组会\3Dtruss大变形\材料拟合\LC_20260615\20260615LC\spec608\specimen.dat`(主)+ `spec609`(备)
- **截面**:10.4 × 8.7 mm² = 90.48 mm²
- **真应力 csv**:`D:\组会\3Dtruss大变形\材料拟合\LC_20260615\spec608_matA_10.4x8.7.csv`
- **拟合脚本**:`D:\组会\3Dtruss大变形\材料拟合\LC_20260615\_jc_damage_matA.py`
- **拟合 log / 图**:`_jc_damage_matA.log` / `matA_JCdamage.png`(同目录)
- **拟合参数**:E=**448.471**, ν=**0.40**, A=**12.4393**, B=**262.0127**, n=**2.5640**, ε_pd=**0.0280**, u_f=**0.6475**  (RMS=0.1911 MPa,1.30% of peak 14.72)

### Tough 2000 v1.1(LC_20260618)

- **原始 MTS .dat**:`D:\组会\3Dtruss大变形\材料拟合\LC_20260618\20260618LC\spec604\specimen.dat` + `spec605\specimen.dat`(联合拟,弃 602/603/606/607)
- **截面**:8.9 × 3.33 mm² = 29.637 mm²
- **WeChat zip 源**:`D:\downloads\WeChat\xwechat_files\wxid_9q7wtz4tme1j22_8905\temp\RWTemp\2026-06\21f4ce7259e8caed7c5adaa6f7a856f7\20260618LC.zip`
- **真应力 csv**:`D:\组会\3Dtruss大变形\材料拟合\LC_20260618\dogbone_v1_spec604.csv` + `dogbone_v1_spec605.csv`(4 列:eps_nomi, sig_nomi_MPa, sig_true_MPa, eps_plas)
- **拟合脚本**:`D:\组会\3Dtruss大变形\材料拟合\LC_20260618\_jc_damage_v1.py`
- **拟合 log / 图**:`_jc_damage_v1.log` / `dogbone_v1_jc_fit.png`(同目录)
- **拟合参数**:E=**1321.7**, ν=**0.40**(沿用 v2 默认), A=**30.86**, B=**500.0**(贴上界), n=**2.215**, ε_pd=**0.0332**, u_f=**0.7258**  (RMS=0.6687 MPa,1.67% of peak 39.98)
- **纯 JC(无 damage)4 参数对照**(2026-06-29 增补,不替换):E=**1333.5**, A=**30.585**, B=**311.8**, n=**2.7993**(RMS=0.7264 MPa,1.82%);5 seeds 同解、无参数贴界,略差于带 damage 版,带 damage 6 参数仍为权威。脚本/图:`D:\组会\3Dtruss大变形\材料拟合\LC_20260618\_jc_nodamage_v1.py` / `dogbone_v1_nodamage_jc_fit.png`

### v2 vs v1.1 差异

| 项 | v2 | v1.1 | 倍数 |
|---|---|---|---|
| E (MPa) | 448 | 1322 | ×3.0 |
| Peak σ_true (MPa) | 14.7 | 40.0 | ×2.7 |
| max ε_true | 0.38 | 0.32 | ≈ |

**当前 14664**(14661 重启版):用 **v2** THETA0(2026-06-22 用户确认 4 case 全 v2 印制)。换 v1.1 印的 lattice 必须切 v1.1 6 参数。

### case → 材料权威对照表(2026-06-22)

| case | lattice txt | TPU 材料 |
|---|---|---|
| P222_38429 | `/Group/P222/38429.txt` | **v2** |
| Pbar4m2_4799 | `/Group/Pbar4m2/4799.txt` | **v2** |
| I422_7871 | `/Group/I422/7871.txt` | **v2** |
| P422_15689 | `/Group/P422/15689.txt` | **v2** |

权威 + 命名规则:`D:\组会\3Dtruss大变形\CLAUDE.md`。
