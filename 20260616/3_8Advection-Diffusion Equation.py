"""
一维稳态对流扩散方程有限元求解与稳定化
=====================================
求解方程： v * dtheta/dx - kappa * d2theta/dx2 = 0,  x in [0, L]
边界条件： theta(0) = 0, theta(L) = 1
比较三种格式：标准 Galerkin (alpha=0)、迎风格式 (alpha=1)、SUPG/Petrov-Galerkin (alpha=alpha_opt)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- 辅助函数 ----------
def alpha_supg(Pe):
    """
    计算 SUPG 最优 alpha 参数： alpha = coth(Pe) - 1/Pe
    当 Pe 很小时，使用级数展开避免除零，返回 0
    """
    if abs(Pe) < 1e-8:
        return 0.0
    # 使用稳定计算：coth(x) = (exp(2x)+1)/(exp(2x)-1)
    # 对于小 Pe，直接用级数展开：coth(x) = 1/x + x/3 - x^3/45 + ...
    if abs(Pe) < 0.1:
        return Pe / 3.0 - Pe**3 / 45.0   # coth - 1/Pe ≈ Pe/3 - Pe^3/45
    else:
        # 直接用公式，避免 exp 溢出，使用 expm1
        exp_2p = np.expm1(2 * Pe)        # exp(2Pe) - 1
        if exp_2p == 0:
            return 0.0
        coth = (np.exp(2 * Pe) + 1) / (np.exp(2 * Pe) - 1)
        # 等价于 (exp(2Pe)+1)/(exp(2Pe)-1)，但 exp(2Pe) 可能很大，使用 expm1 更稳定：
        # coth = (exp(2Pe)+1)/(exp(2Pe)-1) = 1 + 2/(exp(2Pe)-1)
        # 所以 alpha = coth - 1/Pe = 1 + 2/(exp(2Pe)-1) - 1/Pe
        # 但直接使用 np.exp 可能溢出，所以采用上述转换：
        coth_stable = 1.0 + 2.0 / np.expm1(2 * Pe)   # 当 Pe 较大时，exp(2Pe) 可能溢出，但 expm1 处理了
        return coth_stable - 1.0 / Pe


def element_matrix(kappa, v, le, alpha):
    """
    构造两节点线性单元的 2x2 对流扩散单元矩阵（稳定化形式）
    输入：
        kappa : 扩散系数
        v     : 对流速度（标量，正数）
        le    : 单元长度
        alpha : 稳定化参数（0 为 Galerkin，1 为迎风，opt 为 SUPG）
    返回：
        2x2 numpy 数组
    """
    # 人工扩散系数
    kappa_bar = kappa + alpha * v * le / 2.0
    # 扩散部分
    K_diff = (kappa_bar / le) * np.array([[1, -1], [-1, 1]])
    # 对流部分（中心差分近似积分）
    K_conv = (v / 2.0) * np.array([[-1, 1], [-1, 1]])
    return K_diff + K_conv


def solve_advection_diffusion(nel, L, v, kappa, alpha):
    """
    求解一维对流扩散方程，返回节点坐标、数值解、精确解
    输入：
        nel   : 单元数
        L     : 区间长度
        v     : 对流速度
        kappa : 扩散系数
        alpha : 稳定化参数
    返回：
        x     : 节点坐标数组 (nel+1,)
        theta : 数值解数组 (nel+1,)
        theta_exact : 精确解数组 (nel+1,)
    """
    nnode = nel + 1
    le = L / nel
    # 初始化总体矩阵和右端向量
    K = np.zeros((nnode, nnode))
    F = np.zeros(nnode)

    # 组装每个单元
    for e in range(nel):
        # 单元局部节点编号 (全局)
        i = e
        j = e + 1
        # 计算单元矩阵
        Ke = element_matrix(kappa, v, le, alpha)
        # 组装到总体矩阵
        K[i, i] += Ke[0, 0]
        K[i, j] += Ke[0, 1]
        K[j, i] += Ke[1, 0]
        K[j, j] += Ke[1, 1]

    # 施加 Dirichlet 边界条件 (置大数法，或直接划行划列)
    # 边界 theta(0) = 0, theta(L) = 1
    # 为了保留矩阵性质分析，我们将边界条件后的矩阵副本用于求解
    K_bc = K.copy()
    F_bc = F.copy()

    # 左端点 (节点0) 固定为 0
    K_bc[0, :] = 0.0
    K_bc[0, 0] = 1.0
    F_bc[0] = 0.0

    # 右端点 (节点 nnode-1) 固定为 1
    K_bc[-1, :] = 0.0
    K_bc[-1, -1] = 1.0
    F_bc[-1] = 1.0

    # 求解线性方程组
    theta = np.linalg.solve(K_bc, F_bc)

    # 计算精确解（稳定形式，避免指数溢出）
    # theta_exact(x) = (exp(v*x/kappa) - 1) / (exp(v*L/kappa) - 1)
    # 当 v*L/kappa 很大时，使用 (1 - exp(-v*x/kappa)) / (1 - exp(-v*L/kappa))
    x = np.linspace(0, L, nnode)
    # 使用稳定表达式
    if v * L / kappa > 700:  # 防止 exp 溢出
        # 用负指数形式
        numerator = 1.0 - np.exp(-v * x / kappa)
        denominator = 1.0 - np.exp(-v * L / kappa)
    else:
        numerator = np.exp(v * x / kappa) - 1.0
        denominator = np.exp(v * L / kappa) - 1.0
    theta_exact = numerator / denominator

    return x, theta, theta_exact


def compute_error(theta_num, theta_exact):
    """计算最大节点误差"""
    return np.max(np.abs(theta_num - theta_exact))


def analyze_matrix(K, title="总体矩阵"):
    """
    分析矩阵的对称性和正定性
    输出矩阵基本信息、是否对称、特征值范围
    """
    print("\n" + "="*60)
    print(f"矩阵分析: {title}")
    print(f"矩阵形状: {K.shape}")
    # 对称性检查
    sym_diff = np.linalg.norm(K - K.T)
    is_sym = np.allclose(K, K.T)
    print(f"对称性: {'是' if is_sym else '否'} (Frobenius 范数差 = {sym_diff:.2e})")

    # 正定性检查：计算特征值
    try:
        eigvals = np.linalg.eigvalsh(K) if is_sym else np.linalg.eigvals(K)
        # 取实部最小的几个
        eig_vals_real = np.real(eigvals)
        min_eig = np.min(eig_vals_real)
        max_eig = np.max(eig_vals_real)
        print(f"特征值范围: [{min_eig:.4e}, {max_eig:.4e}]")
        if min_eig > 0:
            print("矩阵是正定的 (所有特征值 > 0)")
        elif min_eig == 0:
            print("矩阵是半正定的 (最小特征值 ≈ 0)")
        else:
            print(f"矩阵不是正定的 (最小特征值 = {min_eig:.4e} < 0)")
    except Exception as e:
        print("特征值计算失败:", e)


# ---------- 主程序 ----------
def main():
    # 固定参数
    L = 1.0
    v = 1.0
    nel = 20          # 单元数
    Peclet_values = [0.1, 3.0]   # 要测试的 Pe

    # 存储结果用于绘图
    results = {}

    for Pe in Peclet_values:
        print(f"\n{'='*60}")
        print(f"计算 Peclet 数 Pe = {Pe}")
        # 根据 Pe 计算 kappa
        le = L / nel
        kappa = v * le / (2.0 * Pe)
        print(f"  le = {le:.4f}, kappa = {kappa:.6f}")

        # 三种 alpha
        alpha_values = [0.0, 1.0, None]   # None 表示使用 SUPG 最优
        alpha_names = ['Galerkin (alpha=0)', 'Upwind (alpha=1)', 'SUPG (alpha_opt)']

        # 存储当前 Pe 下的解
        results[Pe] = {}
        errors = []

        for idx, alpha in enumerate(alpha_values):
            if alpha is None:
                # 计算 SUPG 最优 alpha
                # 注意：Pe 为单元 Peclet 数，但公式中 alpha_opt = coth(Pe) - 1/Pe
                # 这里 Pe 已经定义好了，直接使用
                alpha_used = alpha_supg(Pe)
                name = f'SUPG (alpha={alpha_used:.4f})'
            else:
                alpha_used = alpha
                name = alpha_names[idx]

            x, theta, theta_exact = solve_advection_diffusion(nel, L, v, kappa, alpha_used)
            err = compute_error(theta, theta_exact)
            errors.append(err)
            # 保存结果
            if Pe not in results:
                results[Pe] = {}
            results[Pe][name] = {'x': x, 'theta': theta, 'error': err}

            # 打印误差
            print(f"  {name:25s} 最大节点误差 = {err:.4e}")

        # 保存精确解（所有方法共用）
        # 用任意一个方法的 x 即可，但精确解相同，这里用 Galerkin (alpha=0) 的结果
        x_exact = results[Pe][alpha_names[0]]['x']
        # 重新计算精确解（与前面一致）
        if v * L / kappa > 700:
            theta_exact_vals = (1.0 - np.exp(-v * x_exact / kappa)) / (1.0 - np.exp(-v * L / kappa))
        else:
            theta_exact_vals = (np.exp(v * x_exact / kappa) - 1.0) / (np.exp(v * L / kappa) - 1.0)
        results[Pe]['exact'] = {'x': x_exact, 'theta': theta_exact_vals}

        # 任务4：对 Pe=3.0 输出标准 Galerkin 总体矩阵（施加边界条件前）
        if Pe == 3.0:
            # 重新组装一次标准 Galerkin 的矩阵（不施边界）
            K_orig = np.zeros((nel+1, nel+1))
            le0 = L / nel
            for e in range(nel):
                Ke = element_matrix(kappa, v, le0, 0.0)   # alpha=0
                i = e; j = e+1
                K_orig[i,i] += Ke[0,0]; K_orig[i,j] += Ke[0,1]
                K_orig[j,i] += Ke[1,0]; K_orig[j,j] += Ke[1,1]
            analyze_matrix(K_orig, "标准 Galerkin 总体矩阵 (未施加边界条件)")

            # 构造施加边界后的矩阵用于对比
            K_bc = K_orig.copy()
            K_bc[0, :] = 0; K_bc[0,0] = 1
            K_bc[-1,:] = 0; K_bc[-1,-1] = 1
            analyze_matrix(K_bc, "标准 Galerkin 总体矩阵 (施加边界条件后)")

        # 绘图 (每个 Pe 一张图)
        fig, ax = plt.subplots(figsize=(8, 5))
        # 精确解
        ax.plot(results[Pe]['exact']['x'], results[Pe]['exact']['theta'],
                'k-', linewidth=2, label='Exact')
        # 三种格式
        for name in results[Pe].keys():
            if name == 'exact':
                continue
            data = results[Pe][name]
            ax.plot(data['x'], data['theta'], 'o-', label=name)
        ax.set_xlabel('x')
        ax.set_ylabel('theta')
        ax.set_title(f'Pe = {Pe}  (nel={nel}, v={v}, kappa={kappa:.4f})')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        # 保存图片
        plt.savefig(f'Pe_{Pe:.1f}_comparison.png', dpi=150)
        plt.show()

    print("\n程序执行完毕。图片已保存为 Pe_0.1_comparison.png 和 Pe_3.0_comparison.png")

    # ---------- 网格加密收敛性 ----------
    # 基于 Pe=3.0, nel=20 的 kappa 作为固定物理参数
    # 直接使用之前计算得到的 kappa（Pe=3.0 时）
    le_ref = L / 20
    kappa_fixed = v * le_ref / (2 * 3.0)  # 对应 Pe=3.0
    nel_list = [10, 20, 40, 80]
    convergence_study(L, v, kappa_fixed, nel_list, Pe_ref=3.0)

def convergence_study(L, v, kappa, nel_list, Pe_ref=3.0):
    """
    网格加密收敛性研究
    固定物理参数 v, kappa, L，改变单元数 nel，计算 Galerkin 和 SUPG 的误差，
    绘制误差随单元数（或单元长度）的收敛曲线。

    输入：
        L       : 区间长度
        v       : 对流速度
        kappa   : 扩散系数（固定值）
        nel_list: 单元数列表，如 [10, 20, 40, 80]
        Pe_ref  : 仅用于打印参考信息
    """
    print("\n" + "=" * 60)
    print("网格加密收敛性研究")
    print(f"固定参数: L={L}, v={v}, kappa={kappa:.6f}")
    print(f"参考单元 Peclet (nel=20): Pe = {v * (L / 20) / (2 * kappa):.3f}")

    errors = {'Galerkin': [], 'SUPG': []}
    nel_used = []
    le_list = []

    for nel in nel_list:
        le = L / nel
        Pe = v * le / (2 * kappa)  # 当前网格下的单元 Peclet 数
        print(f"\nnel = {nel:3d}, le = {le:.4f}, Pe = {Pe:.4f}")

        # 计算 Galerkin (alpha=0)
        _, theta_gal, theta_ex = solve_advection_diffusion(nel, L, v, kappa, 0.0)
        err_gal = compute_error(theta_gal, theta_ex)
        errors['Galerkin'].append(err_gal)

        # 计算 SUPG (alpha_opt)
        alpha_opt = alpha_supg(Pe)
        _, theta_supg, _ = solve_advection_diffusion(nel, L, v, kappa, alpha_opt)
        err_supg = compute_error(theta_supg, theta_ex)
        errors['SUPG'].append(err_supg)

        print(f"  Galerkin 误差: {err_gal:.4e}")
        print(f"  SUPG     误差: {err_supg:.4e}")

        nel_used.append(nel)
        le_list.append(le)

    # 输出误差汇总表
    print("\n误差汇总表 (收敛性研究)")
    print("-" * 60)
    print(f"{'nel':>5}  {'le':>8}  {'Galerkin 误差':>16}  {'SUPG 误差':>16}")
    print("-" * 60)
    for nel, le, err_g, err_s in zip(nel_used, le_list, errors['Galerkin'], errors['SUPG']):
        print(f"{nel:5d}  {le:8.4f}  {err_g:16.4e}  {err_s:16.4e}")
    print("-" * 60 + "\n")

    # ---- 绘制收敛曲线（横轴为 le） ----
    fig, ax = plt.subplots(figsize=(8, 5))
    # 横轴使用 le（对数坐标）
    ax.loglog(le_list, errors['Galerkin'], 'o-', label='Galerkin (α=0)', color='red')
    ax.loglog(le_list, errors['SUPG'], 's-', label='SUPG (α=α_opt)', color='blue')

    # 参考一阶斜率线
    le_ref = le_list[0]
    err_ref = errors['Galerkin'][0]
    le_line = np.array([le_list[0], le_list[-1]])
    ax.loglog(le_line, err_ref * (le_line / le_ref) ** 1, 'k--', label='O(le) reference', alpha=0.6)

    ax.set_xlabel('Element length le = L/nel')
    ax.set_ylabel('Maximum nodal error')
    ax.set_title(f'Convergence study (v={v}, κ={kappa:.4f})')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('convergence_study.png', dpi=150)
    plt.show()

    print("\n误差收敛曲线已保存为 convergence_study.png")

if __name__ == "__main__":
    main()

