import numpy as np
import json
import time
import scipy as sp
import matplotlib.pyplot as plt
from truss2_3 import print_matrix, generate_LM, postprocess_elements
from pypardiso import spsolve

# 使用微软雅黑（支持更多 Unicode 字符）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置数学字体为默认（避免上标等符号缺失）
plt.rcParams['mathtext.default'] = 'regular'
# ------------------------------------------------------------
# 1. 读取 2.3 导出的 JSON 文件
# ------------------------------------------------------------
def load_reduced_system(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    K_FF = np.array(data['K_FF'])
    rhs = np.array(data['rhs'])
    free_dofs = data['free_dofs']
    fixed_dofs = data['fixed_dofs']
    full_K = np.array(data.get('full_K', [[]]))
    full_force = np.array(data.get('full_force', []))
    return K_FF, rhs, free_dofs, fixed_dofs, full_K, full_force

# ------------------------------------------------------------
# 2. LDL^T 分解
# ------------------------------------------------------------
def ldlt_factor(K):
    """
    对对称正定矩阵 K 进行 LDL^T 分解，返回 L 和 D。
    K: n x n 对称正定矩阵（仅使用下三角部分，但可传入完整矩阵）
    返回:
        L: n x n 单位下三角矩阵
        D: n 维对角矩阵（一维数组）
    若遇到非正主元，抛出 ValueError。
    """
    n = K.shape[0]

    A = K.astype(float).copy()
    L = np.eye(n)
    D = np.zeros(n)

    for j in range(n):
        # 计算 D[j]
        d = A[j, j]
        for k in range(j):
            d -= L[j, k] ** 2 * D[k]
        if d <= 1e-15:
            raise ValueError(f"矩阵非正定或零主元: 第 {j} 个主元 = {d}")
        D[j] = d

        # 计算 L[i, j] for i > j
        for i in range(j + 1, n):
            l = A[i, j]
            for k in range(j):
                l -= L[i, k] * L[j, k] * D[k]
            L[i, j] = l / D[j]

    return L, D

# ------------------------------------------------------------
# 3. LDL^T 求解 (前代 + 对角 + 回代)
# ------------------------------------------------------------
def ldlt_solve(L, D, R):
    """
    求解 LDL^T a = R
    L: 单位下三角矩阵 (n x n)
    D: 对角矩阵 (一维数组)
    R: 右端项 (n 维)
    返回 a (n 维)
    """
    n = len(R)
    # 前代: L y = R
    y = np.zeros(n)
    for i in range(n):
        s = R[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s
    # 对角求解: D z = y
    z = y / D
    # 回代: L^T a = z
    a = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = z[i]
        for j in range(i + 1, n):
            s -= L[j, i] * a[j]
        a[i] = s
    return a

# ------------------------------------------------------------
# 4. 多载荷工况求解 (演示先分解再求解多个右端项)
# ------------------------------------------------------------
def ldlt_solve_multiple(L, D, R_list):
    """
    利用已分解的 L, D 求解多个右端项，返回解列表。
    参数:
        L, D :  LDL^T 分解结果
        R_list : list of ndarray, 多个右端项向量
    返回:
        solutions : list of ndarray, 对应的解向量列表
    """
    solutions = []
    for R in R_list:
        a = ldlt_solve(L, D, R)
        solutions.append(a)
    return solutions

# ------------------------------------------------------------
# 5. 残差和条件数计算
# ------------------------------------------------------------
def vector_norm(v):
    """计算向量 v 的欧几里得范数 (sqrt(Σ v_i²))"""
    return np.sqrt(np.sum(v * v))

def residual_norm(K, a, R):
    """计算绝对残差范数 ||R - K a||"""
    r = R - K @ a
    return vector_norm(r)

def condition_number(K):
    """
    针对 2×2 对称正定矩阵手动计算条件数（最大特征值 / 最小特征值）
    对于非 2×2 矩阵，返回 None 并打印提示
    """
    if K.shape != (2, 2):
        print("警告: condition_number_2x2 仅支持 2×2 矩阵，返回 None")
        return None
    a, b = K[0, 0], K[0, 1]
    c, d = K[1, 0], K[1, 1]
    # 对称性检查
    if abs(b - K[1, 0]) > 1e-12:
        raise ValueError("矩阵不对称")
    # 特征值公式: λ = (a+d ± sqrt((a-d)² + 4b²)) / 2
    trace = a + d
    disc = (a - d) ** 2 + 4 * b * b
    sqrt_disc = np.sqrt(disc)   # np.sqrt 允许，不调用 linalg
    lambda1 = (trace + sqrt_disc) / 2.0
    lambda2 = (trace - sqrt_disc) / 2.0
    if lambda2 <= 0:
        raise ValueError("矩阵非正定")
    return lambda1 / lambda2

# ------------------------------------------------------------
# 6. 算例0：使用作业 2.3 导出的文件进行求解
# ------------------------------------------------------------
# 在文件开头添加导入（如果可用）
try:
    from truss2_3 import postprocess_elements

    USE_23_MODULE = True
except ImportError:
    USE_23_MODULE = False
    print("警告: 无法导入 truss2_3 模块，将无法计算单元轴力。")


def example1_integration():
    """算例1：一维两单元杆结构，与2.3作业衔接"""
    json_file = "D:/application/Git/FEM-BOOK/20260609/算例1.json"
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    K_FF = np.array(data['K_FF'])
    rhs = np.array(data['rhs'])
    free_dofs = data['free_dofs']
    fixed_dofs = data['fixed_dofs']
    full_K = np.array(data.get('full_K', [[]]))
    full_force = np.array(data.get('full_force', []))
    coords = data.get('coords')
    IEN = data.get('IEN')
    ndof = data.get('ndof', 2)
    k_list = data.get('k')
    E_list = data.get('E')
    A_list = data.get('CArea')

    print("\n=== 算例0-1：复用2.3作业的桁架模型（一维两单元杆结构） ===")
    print("缩减方程规模: ", K_FF.shape)

    # 条件数（手写2x2）
    if K_FF.shape == (2, 2):
        cond_val = condition_number(K_FF)
        print(f"条件数: {cond_val:.6f}")

    # 1. 使用本作业的 LDL^T 求解器求解缩减方程
    print("\n1. 使用本作业的 LDL^T 求解器求解缩减方程")
    try:
        L, D = ldlt_factor(K_FF.copy())
        d_F = ldlt_solve(L, D, rhs)
        print(f"求解结果 (自由部分): {d_F}")

        if full_K.size > 0:
            neq = len(full_force)
            d_full = np.zeros(neq)
            d_full[fixed_dofs] = 0.0
            d_full[free_dofs] = d_F

            # 节点1 x方向约束反力
            r = full_K @ d_full - full_force
            if len(r) > 0:
                r1x = r[0]
                print("\n2. 使用作业2.3后处理模块计算节点1的约束反力")
                print(f"节点1 x方向约束反力: {r1x:.6f} ")

            # 计算单元轴力
            if coords and IEN:
                from truss2_3 import postprocess_elements
                if k_list is not None:
                    elem_results = postprocess_elements(coords, IEN, None, None, d_full, ndof, k_list=k_list)
                elif E_list is not None and A_list is not None:
                    elem_results = postprocess_elements(coords, IEN, E_list, A_list, d_full, ndof)
                else:
                    print("无法计算单元轴力：缺少模型数据")
                    return
                print("\n3. 使用作业2.3后处理模块计算各单元轴力:")
                for e, (L, c, s, stress, N) in enumerate(elem_results):
                    print(f"单元{e + 1}: 轴力 = {N:.6f}")
    except ValueError as e:
        print("LDL^T 分解失败:", e)


def example2_integration():
    """算例2：二维两杆桁架结构，与2.3作业衔接"""
    json_file = "D:/application/Git/FEM-BOOK/20260609/算例2.json"
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    K_FF = np.array(data['K_FF'])
    rhs = np.array(data['rhs'])
    free_dofs = data['free_dofs']
    fixed_dofs = data['fixed_dofs']
    full_K = np.array(data.get('full_K', [[]]))
    full_force = np.array(data.get('full_force', []))
    coords = data.get('coords')
    IEN = data.get('IEN')
    ndof = data.get('ndof', 2)
    E_list = data.get('E')
    A_list = data.get('CArea')

    print("\n=== 算例0-2：复用2.3作业的桁架模型（二维两杆桁架结构） ===")
    print("缩减方程规模: ", K_FF.shape)

    # 条件数
    if K_FF.shape == (2, 2):
        cond_val = condition_number(K_FF)
        print(f"条件数: {cond_val:.6f}")

    # 1. 使用2.3作业代码生成LM、总体刚度矩阵 K 和缩减矩阵 K_FF
    print("\n1. 使用2.3作业的代码生成矩阵")
    if full_K.size > 0:
        print("\n总体刚度矩阵 K (full_K)：")
        print_matrix("full_K", full_K, "%12.6f")
    else:
        print("总体刚度矩阵 K 未在JSON中提供。")
    print("\n缩减矩阵 K_FF：")
    print_matrix("K_FF", K_FF, "%12.6f")

    if IEN is not None and ndof:
        from truss2_3 import generate_LM  # 复用2.3的生成函数
        LM = generate_LM(np.array(IEN), ndof)
        print("\nLM矩阵 (对号矩阵):")
        print(LM)
    else:
        print("警告：JSON中缺少IEN或ndof，无法生成LM矩阵。")

    # 2. 节点3位移
    try:
        L, D = ldlt_factor(K_FF.copy())
        d_F = ldlt_solve(L, D, rhs)

        print("\n2. 使用本作业的求解器求解节点3位移:")
        print(f"u3 = {d_F[0]:.6f}, v3 = {d_F[1]:.6f}")

        if full_K.size > 0:
            neq = len(full_force)
            d_full = np.zeros(neq)
            d_full[fixed_dofs] = 0.0
            d_full[free_dofs] = d_F

            # 3. 回到2.3后处理模块，求解单元1、2的应力
            if coords and IEN and E_list is not None and A_list is not None:
                from truss2_3 import postprocess_elements
                elem_results = postprocess_elements(coords, IEN, E_list, A_list, d_full, ndof)
                print("\n3. 使用作业2.3后处理模块计算单元应力:")
                for e, (L, c, s, stress, N) in enumerate(elem_results):
                    print(f"单元{e + 1}: 应力 = {stress:.6f}")
            else:
                print("无法计算单元应力：缺少模型数据")
    except ValueError as e:
        print("LDL^T 分解失败:", e)

# ------------------------------------------------------------
# 7. 算例1: 三对角对称正定矩阵性能测试
# ------------------------------------------------------------
def test_tridiagonal():
    """生成三对角矩阵，测试不同规模下的求解时间"""
    sizes = [10, 100, 500, 1000]
    print("\n=== 算例1：三对角对称正定矩阵性能测试 ===")
    times = []
    errors = []
    for n in sizes:
        # 构造三对角矩阵
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 2.0
            if i > 0:
                K[i, i - 1] = -1.0
            if i < n - 1:
                K[i, i + 1] = -1.0
        exact = np.ones(n)
        R = K @ exact

        start = time.perf_counter()
        try:
            L, D = ldlt_factor(K.copy())
            a = ldlt_solve(L, D, R)
            elapsed = time.perf_counter() - start
            max_err = np.max(np.abs(a - exact))
            times.append(elapsed)
            errors.append(max_err)
            print(f"n={n:4d}, 时间={elapsed:.6f}s, 最大误差={max_err:.2e}")
        except ValueError as e:
            print(f"n={n}: 分解失败 - {e}")
            times.append(None)
            errors.append(None)

    # 绘制求解时间随 n 的变化（双对数坐标）
    valid = [(s, t) for s, t in zip(sizes, times) if t is not None]
    if valid:
        n_vals = [v[0] for v in valid]
        t_vals = [v[1] for v in valid]
        plt.figure(figsize=(8, 5))
        plt.loglog(n_vals, t_vals, 'o-', label='实测时间')
        # 绘制 O(n^3) 参考线（使用第一个点确定比例）
        ref_n = n_vals[0]
        ref_t = t_vals[0]
        n_ref = np.array(n_vals)
        t_ref = ref_t * (n_ref / ref_n) ** 3
        plt.loglog(n_ref, t_ref, '--', color='red', label='O(n³) 参考线')
        plt.xlabel('矩阵阶数 n')
        plt.ylabel('求解时间 (s)')
        plt.title('稠密 LDL$^T$ 求解时间随 n 增长趋势')
        plt.legend()
        plt.grid(True)
        plt.savefig('tridiagonal_performance.png', dpi=150)
        plt.close()
        print("性能曲线已保存为 tridiagonal_performance.png")


# ------------------------------------------------------------
# 8. 算例2: 非正定矩阵检测
# ------------------------------------------------------------
def test_non_positive():
    """测试非正定矩阵，验证分解函数能否正确报错"""
    print("\n=== 算例2：非正定矩阵检测 ===")
    K = np.array([[1.0, 2.0], [2.0, 1.0]])
    print("测试矩阵:\n", K)
    try:
        L, D = ldlt_factor(K.copy())
        print("错误: 分解成功，但该矩阵非正定")
    except ValueError as e:
        print(f"正确检测到非正定: {e}")


# ------------------------------------------------------------
# 9. 任务2: 病态矩阵误差分析 (双精度 vs 低精度模拟)
# ------------------------------------------------------------
def vector_norm(v):
    """手动计算向量欧几里得范数"""
    return np.sqrt(np.sum(v * v))

def test_ill_conditioned():
    print("\n=== 任务2：病态矩阵误差分析 ===")
    # 原始高精度矩阵
    K = np.array([[1.0, 1.0], [1.0, 1.0001]], dtype=float)
    a_exact = np.array([1.0, 1.0])
    R = K @ a_exact

    print("\n原始数据（双精度）:")
    print(f"K =\n{K}")
    print(f"R = {R}")

    # ---- 双精度求解 ----
    print("\n【双精度计算】")
    try:
        K_copy = K.copy()
        L, D = ldlt_factor(K_copy)
        a_double = ldlt_solve(L, D, R)
        print(f"1. 数值解 a = {a_double}")
        r_double = R - K @ a_double
        print(f"2. 残差 r = {r_double}")
        rel_res_double = vector_norm(r_double) / (vector_norm(R) + 1e-15)
        print(f"3. 相对残差 = {rel_res_double:.2e}")
        rel_err_double = vector_norm(a_double - a_exact) / (vector_norm(a_exact) + 1e-15)
        print(f"4. 相对误差 = {rel_err_double:.2e}")
    except ValueError as e:
        print("双精度分解失败:", e)
    cond_double = np.linalg.cond(K)
    print(f"5.条件数= {cond_double:.2f}")

    # ---- 低精度模拟（单精度 float32）----
    print("\n【低精度计算 (单精度 float32，约7位有效数字)】")
    K_low = K.astype(np.float32).astype(np.float64)  # 转为 float32 再转回 float64 模拟
    R_low = R.astype(np.float32).astype(np.float64)
    print("低精度数据 (float32 存储后):")
    print(f"K_low =\n{K_low}")
    print(f"R_low = {R_low}")
    try:
        K_low_copy = K_low.copy()
        L_low, D_low = ldlt_factor(K_low_copy)
        a_low = ldlt_solve(L_low, D_low, R_low)
        print(f"1. 数值解 a = {a_low}")
        r_low = R_low - K_low @ a_low
        print(f"2. 残差 r = {r_low}")
        rel_res_low = vector_norm(r_low) / (vector_norm(R_low) + 1e-15)
        print(f"3. 相对残差 = {rel_res_low:.2e}")
        rel_err_low = vector_norm(a_low - a_exact) / (vector_norm(a_exact) + 1e-15)
        print(f"4. 相对误差 = {rel_err_low:.2e}")
    except ValueError as e:
        print("低精度分解失败:", e)
    cond_low = np.linalg.cond(K_low)
    print(f"5.条件数= {cond_low:.2f}")


# ------------------------------------------------------------
# 10. 多载荷工况演示
# ------------------------------------------------------------
def test_multiple_rhs():
    """演示先分解再求解多个右端项的效率优势"""
    print("\n=== 多载荷工况效率分析 ===")
    # 使用一个较小的三对角矩阵
    n = 100
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] = 2.0
        if i > 0:
            K[i, i - 1] = -1.0
        if i < n - 1:
            K[i, i + 1] = -1.0

    # 构造多个右端项 (例如5个)
    R_list = []
    for k in range(5):
        exact = np.full(n, k + 1)  # 不同常数向量
        R_list.append(K @ exact)

    # 先分解
    start_factor = time.perf_counter()
    L, D = ldlt_factor(K.copy())
    factor_time = time.perf_counter() - start_factor

    # 再逐个求解
    start_solve = time.perf_counter()
    solutions = ldlt_solve_multiple(L, D, R_list)
    solve_time = time.perf_counter() - start_solve

    print(f"矩阵规模 n={n}")
    print(f"分解时间: {factor_time:.6f}s")
    print(f"求解5个右端项总时间: {solve_time:.6f}s")
    print(f"平均每个右端项求解时间: {solve_time / 5:.6f}s")
    # 验证第一个解
    exact_first = np.full(n, 1.0)
    err = np.max(np.abs(solutions[0] - exact_first))
    print(f"第一个右端项解的最大误差: {err:.2e}")

# ------------------------------------------------------------
# 11. 大规模稀疏方程组求解器调用
# ------------------------------------------------------------
def test_sparse_solver():
    n = 5000  # 矩阵阶数（可调整）
    print(f"\n=== 任务3：大规模稀疏方程组求解器调用 (n={n}) ===")

    # 构造三对角矩阵
    diag = 2.0 * np.ones(n)
    off_diag = -1.0 * np.ones(n - 1)
    K = sp.diags([off_diag, diag, off_diag], offsets=[-1, 0, 1], format='csr')

    # 精确解为全1，构造右端项
    a_exact = np.ones(n)
    R = K @ a_exact

    print(f"矩阵阶数: {n}")
    print(f"非零元个数: {K.nnz}")
    mem_kb = (K.data.nbytes + K.indptr.nbytes + K.indices.nbytes) / 1024
    print(f"CSR 存储内存: {mem_kb:.2f} KB")
    print(f"等效稠密存储内存: {n * n * 8 / (1024 ** 2):.2f} MB")

    solver_name = "Intel MKL PARDISO (via pypardiso)"
    start = time.time()
    a_num = spsolve(K, R)
    elapsed = time.time() - start
    print(f"求解器: {solver_name}")
    print(f"求解时间: {elapsed:.4f} s")

    # 计算残差和误差
    r = R - K @ a_num
    rel_res = np.linalg.norm(r) / (np.linalg.norm(R) + 1e-15)
    rel_err = np.linalg.norm(a_num - a_exact) / np.linalg.norm(a_exact)
    print(f"相对残差: {rel_res:.2e}")
    print(f"相对误差: {rel_err:.2e}")

# ------------------------------------------------------------
# 12. 算例4：大规模二维 Poisson 方程有限元求解
# ------------------------------------------------------------
def poisson_2d_fem(nx, ny):
    """
    使用线性三角形单元求解 -Δu = f，Dirichlet 边界条件 u=0。
    理论解 u_exact = sin(πx) sin(πy)，f = 2π² sin(πx) sin(πy)。
    返回: 误差、求解时间、装配时间等详细信息。
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as sla
    import time
    import matplotlib.pyplot as plt

    start_total = time.perf_counter()

    # ---------- 1. 网格生成 ----------
    start_assemble = time.perf_counter()

    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    n_nodes = n_nodes_x * n_nodes_y
    n_tri = 2 * nx * ny  # 每个矩形分为两个三角形

    x = np.linspace(0, 1, n_nodes_x)
    y = np.linspace(0, 1, n_nodes_y)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T

    def node_id(i, j):
        return i * n_nodes_y + j   # 列优先

    IEN = []
    for i in range(nx):
        for j in range(ny):
            n0 = node_id(i, j)
            n1 = node_id(i+1, j)
            n2 = node_id(i+1, j+1)
            n3 = node_id(i, j+1)
            IEN.append([n0, n1, n2])
            IEN.append([n0, n2, n3])
    IEN = np.array(IEN)

    # ---------- 2. 组装总体矩阵和载荷向量 ----------
    neq = n_nodes
    K = sp.lil_matrix((neq, neq))
    R = np.zeros(neq)

    for e in range(n_tri):
        nodes = IEN[e]
        xy = coords[nodes]
        x1, y1 = xy[0]
        x2, y2 = xy[1]
        x3, y3 = xy[2]
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        if area < 1e-12:
            continue
        B = np.array([[y2-y3, y3-y1, y1-y2],
                      [x3-x2, x1-x3, x2-x1]]) / (2*area)
        Ke = area * (B.T @ B)
        xc = (x1+x2+x3)/3
        yc = (y1+y2+y3)/3
        f_val = 2 * np.pi**2 * np.sin(np.pi*xc) * np.sin(np.pi*yc)
        Re = np.full(3, f_val * area / 3)
        for a in range(3):
            i = nodes[a]
            for b in range(3):
                j = nodes[b]
                K[i, j] += Ke[a, b]
            R[i] += Re[a]

    # 保存完整矩阵（用于残差计算）
    K_full = K.tocsr()   # 转为 CSR 便于乘法
    R_full = R.copy()
    assemble_time = time.perf_counter() - start_assemble

    # ---------- 3. 边界条件处理 ----------
    start_bc = time.perf_counter()
    fixed_dofs = []
    for i in range(n_nodes_x):
        for j in range(n_nodes_y):
            if i == 0 or i == n_nodes_x-1 or j == 0 or j == n_nodes_y-1:
                fixed_dofs.append(node_id(i, j))
    fixed_dofs = np.unique(fixed_dofs)
    free_dofs = np.setdiff1d(np.arange(neq), fixed_dofs)

    K_FF = K_full[free_dofs, :][:, free_dofs]
    R_F = R[free_dofs]
    bc_time = time.perf_counter() - start_bc

    # ---------- 4. 求解 ----------
    start_solve = time.perf_counter()
    solver_name = "Intel MKL PARDISO (via pypardiso)"
    a_F = spsolve(K_FF, R_F)
    solve_time = time.perf_counter() - start_solve

    # 重构完整解
    a = np.zeros(neq)
    a[free_dofs] = a_F
    a[fixed_dofs] = 0.0

    total_time = time.perf_counter() - start_total

    # ---------- 5. 计算误差和残差 ----------
    u_exact = np.sin(np.pi * X.ravel()) * np.sin(np.pi * Y.ravel())
    max_err = np.max(np.abs(a - u_exact))
    L2_err = np.sqrt(np.sum((a - u_exact)**2) / np.sum(u_exact**2))

    # 相对残差 ||R - K*a|| / ||R||
    r = R_full - K_full @ a
    rel_res = np.linalg.norm(r) / (np.linalg.norm(R_full) + 1e-15)

    # 输出信息
    print(f"\n=== 算例4：大规模二维Poisson方程有限元求解 (nx={nx}, ny={ny}) ===")
    print(f"单元类型: 线性三角形单元 (T3)")
    print(f"节点数: {neq}")
    print(f"三角形单元数: {n_tri}")
    print(f"未知自由度数: {len(free_dofs)}")
    print(f"总体矩阵非零元个数: {K_full.nnz}")
    print(f"装配时间: {assemble_time:.4f} s")
    print(f"边界条件处理时间: {bc_time:.4f} s")
    print(f"求解时间: {solve_time:.4f} s")
    print(f"总时间: {total_time:.4f} s")
    print(f"求解器名称: {solver_name}")
    print(f"相对残差: {rel_res:.2e}")
    print(f"最大节点误差: {max_err:.4e}")
    print(f"离散 L2 相对误差: {L2_err:.4e}")

    # ---------- 绘制数值解云图和误差云图（分开保存） ----------
    try:
        a_2d = a.reshape(n_nodes_x, n_nodes_y)
        u_exact_2d = u_exact.reshape(n_nodes_x, n_nodes_y)
        err_2d = np.abs(a_2d - u_exact_2d)

        # 1. 数值解云图
        plt.figure(figsize=(6, 5))
        plt.contourf(X, Y, a_2d, levels=20, cmap='viridis')
        plt.colorbar(label='数值解 u_h')
        plt.title(f'数值解 (nx={nx})')
        plt.axis('equal')
        plt.savefig(f'poisson_nx{nx}_solution.png', dpi=150)
        plt.close()
        print(f"数值解云图已保存为 poisson_nx{nx}_solution.png")

        # 2. 误差分布云图
        plt.figure(figsize=(6, 5))
        plt.contourf(X, Y, err_2d, levels=20, cmap='inferno')
        plt.colorbar(label='绝对误差')
        plt.title(f'误差分布 (nx={nx})')
        plt.axis('equal')
        plt.savefig(f'poisson_nx{nx}_error.png', dpi=150)
        plt.close()
        print(f"误差云图已保存为 poisson_nx{nx}_error.png")
    except Exception as e:
        print("绘图失败:", e)

    return max_err, L2_err, assemble_time, solve_time, total_time, rel_res, neq, n_tri, K_full.nnz

# ---------- 调用算例4，测试多个网格 ----------
def test_poisson_convergence():
    """测试 nx=ny=50, 100, 200 的收敛性"""
    print("\n=== 算例4：误差收敛性分析 ===")
    grid_sizes = [(50,50), (100,100), (200,200)]
    results = []
    for nx, ny in grid_sizes:
        max_err, L2_err, ass_t, sol_t, tot_t, rel_res, neq, ntri, nnz = poisson_2d_fem(nx, ny)
        results.append((nx, max_err, L2_err, sol_t, tot_t, rel_res, neq, ntri, nnz))

    print("\n汇总表:")
    print("nx\t节点数\t单元数\t非零元\t求解时间(s)\t总时间(s)\t相对残差\t最大误差\tL2误差")
    for nx, max_err, L2_err, sol_t, tot_t, rel_res, neq, ntri, nnz in results:
        print(f"{nx}\t{neq}\t{ntri}\t{nnz}\t{sol_t:.4f}\t{tot_t:.4f}\t{rel_res:.2e}\t{max_err:.4e}\t{L2_err:.4e}")

    # 绘制误差收敛曲线
    try:
        import matplotlib.pyplot as plt
        # 确保负号正常显示，并启用数学文本
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.default'] = 'regular'

        plt.figure(figsize=(8,5))
        h = [1.0/nx for nx, _,_,_,_,_,_,_,_ in results]
        max_errs = [r[1] for r in results]
        L2_errs = [r[2] for r in results]
        plt.loglog(h, max_errs, 'o-', label='最大误差')
        plt.loglog(h, L2_errs, 's-', label='L2 相关误差')
        # 绘制 O(h^2) 参考线（绿色虚线）
        plt.loglog(h, [h_i**2 for h_i in h], '--', color='green', label='O(h$^2$) 参考线')
        plt.xlabel('网格尺寸 h')
        plt.ylabel('误差')
        plt.legend()
        plt.grid(True)
        plt.title('误差收敛')
        plt.savefig('poisson_convergence.png', dpi=150)
        plt.close()
        print("误差收敛曲线已保存为 poisson_convergence.png")
    except Exception as e:
        print("收敛曲线绘图失败:", e)

        # 绘制求解时间随节点数变化图（验证线性趋势）
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['mathtext.default'] = 'regular'

        neq_list = [r[6] for r in results]
        sol_times = [r[3] for r in results]  # 求解时间

        plt.figure(figsize=(8, 5))
        plt.loglog(neq_list, sol_times, 'o-', label='实测求解时间')
        # 绘制 O(N) 参考线（N为节点数）
        ref_neq = neq_list[0]
        ref_time = sol_times[0]
        neq_ref = np.array(neq_list)
        t_ref = ref_time * (neq_ref / ref_neq)
        plt.loglog(neq_ref, t_ref, '--', color='green', label='O(N) 参考线')
        plt.xlabel('节点数 N')
        plt.ylabel('求解时间 (s)')
        plt.title('稀疏求解器时间随节点数增长趋势')
        plt.legend()
        plt.grid(True)
        plt.savefig('poisson_solve_time.png', dpi=150)
        plt.close()
        print("求解时间曲线已保存为 poisson_solve_time.png")
    except Exception as e:
        print("绘图失败:", e)
# ------------------------------------------------------------
# 主程序入口
# ------------------------------------------------------------
if __name__ == "__main__":
    # 1. 与2.3作业衔接 (需确保存在JSON文件)
    #example1_integration()
    #example2_integration()

    # 2. 运行各验证算例
    test_tridiagonal()
    test_non_positive()
    test_ill_conditioned()
    test_multiple_rhs()
    test_sparse_solver()
    test_poisson_convergence()
    # 打印环境信息
    print("=== 实验环境信息 ===")
    print(f"NumPy 版本: {np.__version__}")
    print(f"SciPy 版本: {sp.__version__}")
    print(f"json 版本: {json.__version__}")
    print(f"Matplotlib 版本: {plt.matplotlib.__version__}")  # 如果已 import matplotlib.pyplot as plt

    try:
        import pypardiso

        print(f"pypardiso 版本: {pypardiso.__version__}")
    except ImportError:
        print("pypardiso 未安装")
