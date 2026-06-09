"""
三维杆单元（空间桁架单元）刚度矩阵与应力计算
功能：
1. 计算单元长度、方向余弦
2. 生成全局坐标系下 6x6 刚度矩阵
3. 根据节点位移计算单元应变、应力、轴力
4. 验证刚度矩阵性质（对称、奇异、半正定、刚体位移、物理意义）
"""

import numpy as np
from fractions import Fraction

def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元的长度、方向余弦和全局刚度矩阵

    参数:
        x1: 节点1坐标，list或array [x, y, z]
        x2: 节点2坐标，list或array [x, y, z]
        E : 弹性模量 (Pa)
        A : 截面积 (m^2)

    返回:
        L          : 单元长度 (m)
        direction  : 元组 (cx, cy, cz) 方向余弦
        Ke         : 6x6 全局刚度矩阵 (ndarray)

    异常:
        ValueError: 当两个节点重合时抛出
    """
    # 转换为 numpy 数组
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    # 计算差值向量和长度
    delta = x2 - x1
    L = np.linalg.norm(delta)

    # 检查退化单元（两节点重合）
    if L < 1e-12:
        raise ValueError("退化单元：两个节点坐标重合，长度为零")

    # 方向余弦
    cx, cy, cz = delta / L
    c = np.array([cx, cy, cz])

    # 局部刚度系数 k = EA/L
    k = E * A / L

    # 计算外积矩阵 (3x3)
    K11 = k * np.outer(c, c)   # 3x3 矩阵

    # 组装 6x6 全局刚度矩阵
    Ke = np.block([[ K11, -K11],
                   [-K11,  K11]])

    return L, (cx, cy, cz), Ke

def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算单元轴向应变、应力和轴力

    参数:
        x1: 节点1坐标 [x, y, z]
        x2: 节点2坐标 [x, y, z]
        E : 弹性模量 (Pa)
        A : 截面积 (m^2)
        de: 节点位移列阵 [u1, v1, w1, u2, v2, w2] (m)

    返回:
        epsilon : 轴向应变 (无量纲)
        sigma   : 轴向应力 (Pa)
        N       : 轴力 (N)，拉力为正
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    de = np.asarray(de, dtype=float)

    # 计算长度和方向余弦
    delta = x2 - x1
    L = np.linalg.norm(delta)
    if L < 1e-12:
        raise ValueError("退化单元：长度为零，无法计算应力")

    c = delta / L   # 方向余弦 (cx, cy, cz)

    # 相对位移向量 (节点2 - 节点1)
    du = de[3:] - de[:3]   # [u2-u1, v2-v1, w2-w1]

    # 轴向伸长 = 相对位移在杆轴方向上的投影
    elong = np.dot(c, du)

    # 应变、应力、轴力
    epsilon = elong / L
    sigma = E * epsilon
    N = sigma * A

    return epsilon, sigma, N

def float_to_fraction(x, tolerance=1e-10):
    """
    将浮点数转换为最简分数（针对有理数）

    参数:
        x: 浮点数
        tolerance: 容差

    返回:
        Fraction 对象
    """
    # 处理接近整数的值
    if abs(x - round(x)) < tolerance:
        return Fraction(int(round(x)), 1)

    # 尝试将浮点数转换为分数（限制分母最大值）
    frac = Fraction(x).limit_denominator(100)

    # 验证转换精度
    if abs(float(frac) - x) < tolerance:
        return frac
    return Fraction(x)

def format_ke_fraction_full(Ke):
    """
    将刚度矩阵以完整的分数形式显示（每个元素都是具体的分数数值）

    参数:
        Ke: 6x6 刚度矩阵（数值）

    返回:
        格式化的字符串
    """
    # 转换为分数
    fractions = []
    for i in range(6):
        row = []
        for j in range(6):
            val = Ke[i, j]
            if abs(val) < 1e-10:
                row.append("0")
            else:
                frac = float_to_fraction(val)
                if frac.denominator == 1:
                    row.append(f"{frac.numerator}")
                else:
                    row.append(f"{frac.numerator}/{frac.denominator}")
        fractions.append(row)

    # 格式化输出
    lines = []
    lines.append("Ke = ")
    for i, row in enumerate(fractions):
        # 计算每列的最大宽度用于对齐
        row_str = "      [" + ", ".join(f"{v:>12}" for v in row) + "]"
        if i < 5:
            row_str += ","
        else:
            row_str += ""
        lines.append(row_str)
    return "\n".join(lines)

def format_direction_cosine_fraction(cx, cy, cz):
    """
    将方向余弦转换为分数形式显示
    """
    frac_cx = float_to_fraction(cx)
    frac_cy = float_to_fraction(cy)
    frac_cz = float_to_fraction(cz)

    return f"({frac_cx.numerator}/{frac_cx.denominator}, {frac_cy.numerator}/{frac_cy.denominator}, {frac_cz.numerator}/{frac_cz.denominator})"

# ======================== 主程序：验证算例与性质验证 ========================
if __name__ == "__main__":

    print("=" * 70)
    print("三维杆单元程序验证")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 算例1：沿 X 轴的一维杆单元
    # ------------------------------------------------------------------
    print("\n【算例1】沿 X 轴的一维杆单元")
    print("-" * 50)

    # 输入数据
    x1_1 = (0.0, 0.0, 0.0)
    x2_1 = (2.0, 0.0, 0.0)
    E1 = 200e9          # 200 GPa
    A1 = 1.0e-4         # 0.0001 m^2
    de1 = [0.0, 0.0, 0.0, 1.0e-3, 0.0, 0.0]   # 节点2 x方向位移1mm

    # 计算刚度矩阵等
    L1, dir1, Ke1 = truss3d_element_stiffness(x1_1, x2_1, E1, A1)
    eps1, sig1, N1 = truss3d_element_stress(x1_1, x2_1, E1, A1, de1)

    print(f"单元长度 L          = {L1:.4f} m  ")
    print(f"方向余弦(cx, cy, cz)= ({dir1[0]:.0f}, {dir1[1]:.0f}, {dir1[2]:.0f}) ")
    # 提取退化的 2x2 子矩阵（对应 u1, u2 自由度）
    k_val = E1 * A1 / L1
    Ke_1d = np.array([[k_val, -k_val],
                      [-k_val, k_val]])
    print("\n退化为一维杆单元形式（只与 x 向自由度有关）:")
    print(Ke_1d)
    print("自由度顺序: [u1, u2]")
    print(f"\n轴向应变 ε       = {eps1:.4e}  ")
    print(f"轴向应力 σ       = {sig1/1e6:.1f} MPa ")
    print(f"轴力 N           = {N1:.1f} N  ")

    # ------------------------------------------------------------------
    # 算例2：空间任意方向杆单元
    # ------------------------------------------------------------------
    print("\n【算例2】空间任意方向杆单元")
    print("-" * 50)

    x1_2 = (0.0, 0.0, 0.0)
    x2_2 = (1.0, 2.0, 2.0)
    E2 = 210e9          # 210 GPa
    A2 = 2.0e-4         # 0.0002 m^2
    de2 = [0.0, 0.0, 0.0, 1.0e-3, 2.0e-3, 2.0e-3]   # 节点2位移 (1,2,2) mm

    L2, dir2, Ke2 = truss3d_element_stiffness(x1_2, x2_2, E2, A2)
    eps2, sig2, N2 = truss3d_element_stress(x1_2, x2_2, E2, A2, de2)

    k_val2 = E2 * A2 / L2

    print(f"单元长度 L          = {L2:.4f} m  ")
    dir_frac_str = format_direction_cosine_fraction(dir2[0], dir2[1], dir2[2])
    print(f"方向余弦(cx, cy, cz)= {dir_frac_str}  ")
    print("\n   刚度矩阵 Ke (分数形式):")
    print(format_ke_fraction_full(Ke2))
    print(f"\n轴向应变 ε          = {eps2:.4e}  ")
    print(f"轴向应力 σ          = {sig2/1e6:.1f} MPa  ")
    print(f"轴力 N             = {N2:.1f} N  ")

    # ------------------------------------------------------------------
    # 性质验证（基于算例2的刚度矩阵）
    # ------------------------------------------------------------------
    print("\n【刚度矩阵性质验证：基于算例2】")
    print("-" * 50)

    # 1. 对称性
    is_sym = np.allclose(Ke2, Ke2.T, rtol=1e-10)
    print(f"1. 对称性: {'满足' if is_sym else '不满足'} (Ke == Ke^T)")

    # 2. 刚体位移检验：整体平移时节点力应为零
    de_rb = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]   # 整体平移
    F_rb = Ke2 @ de_rb
    max_force = np.max(np.abs(F_rb))
    print(f"\n2. 刚体平移产生的最大节点力: {max_force:.2e} N")

    # 3. 特征值分析（半正定性，奇异性）
    eigvals = np.linalg.eigvalsh(Ke2)   # 对称矩阵使用 eigh
    # 由于数值误差，零特征值可能是小量，容差取 1e-8
    zero_tol = 1e-8
    positive_eigs = eigvals[eigvals > zero_tol]
    zero_eigs = eigvals[np.abs(eigvals) <= zero_tol]
    nonzero_eigs = eigvals[np.abs(eigvals) > zero_tol]
    print(f"\n3. 检查Ke的特征值：")
    print(f"   非零特征值个数: {len(nonzero_eigs)}")
    print(f"   非零特征值 = {nonzero_eigs}")
    print(f"   零特征值个数: {len(zero_eigs)} ")
    print(f"   半正定性: {'满足' if np.min(eigvals) >= -1e-8 else '不满足'} (所有特征值 >= 0)")

    # ------------------------------------------------------------------
    # 刚度矩阵物理意义验证：刚度矩阵各列的物理意义：取第 j 列，令单位位移其他零，所得节点力即为该列
    # ------------------------------------------------------------------
    j = 0   # 第一个自由度 (u1)
    de_unit = np.zeros(6)
    de_unit[j] = 1.0
    F_col = Ke2 @ de_unit
    print(f"\n任务4：第 {j+1} 列物理意义验证:")
    print(f"   令 d_{j+1}=1 其余0，计算 Ke * d = {F_col}")
    print(f"   直接取 Ke 的第 {j+1} 列: {Ke2[:, j]}")
    print("   两者一致，说明 Kij 表示第 j 自由度单位位移时在第 i 自由度上需施加的节点力。")

    # 测试退化单元错误处理
    print("\n【退化单元检测】")
    print("-" * 50)
    try:
        L_bad, _, _ = truss3d_element_stiffness((0,0,0), (0,0,0), E1, A1)
        print("未捕获错误，异常处理有问题")
    except ValueError as e:
        print(f"正确捕获错误: {e}")

    print("\n所有验证完成。")