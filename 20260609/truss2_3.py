"""
总体刚度矩阵组装与桁架结构求解
实现二维桁架（含一维杆）的有限元分析流程：
1. 前处理：定义节点、单元、材料、边界、载荷
2. 生成对号矩阵 LM
3. 单元分析：计算单元长度、方向余弦、单元刚度矩阵
4. 直接组装总体刚度矩阵 K
5. 边界处理（缩减法）并求解
6. 后处理：计算单元应力、轴力
"""

import numpy as np
import json

# ------------------------------------------------------------
# 辅助函数：生成对号矩阵 LM
# ------------------------------------------------------------
def generate_LM(IEN, ndof):
    """
    IEN : 单元连接矩阵，形状 (nel, nen)，全局节点编号（从0开始）
    ndof: 每个节点的自由度数
    返回 LM : 形状 (ndof*nen, nel)，每个单元的全局自由度编号（从0开始）
    """
    nel = len(IEN)
    nen = IEN.shape[1] if hasattr(IEN, 'shape') else 2
    LM = np.zeros((ndof * nen, nel), dtype=int)
    for e in range(nel):
        for j in range(nen):
            node = IEN[e][j]  # 全局节点号（0-index）
            for m in range(ndof):
                LM[ndof*j + m, e] = ndof * node + m
    return LM

# ------------------------------------------------------------
# 单元分析：计算单元长度、方向余弦、单元刚度矩阵（二维桁架）
# ------------------------------------------------------------
def element_stiffness_2d(x1, y1, x2, y2, k=None, E=None, A=None):
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    if L < 1e-12:
        raise ValueError("单元长度为零！")
    c = dx / L
    s = dy / L
    if k is None:
        if E is None or A is None:
            raise ValueError("必须提供 k 或 (E, A)")
        k = E * A / L
    # 单元刚度矩阵
    Ke = k * np.array([
        [c*c, c*s, -c*c, -c*s],
        [c*s, s*s, -c*s, -s*s],
        [-c*c, -c*s, c*c, c*s],
        [-c*s, -s*s, c*s, s*s]
    ])
    return Ke, L, (c, s), k

# ------------------------------------------------------------
# 直接组装总体刚度矩阵
# ------------------------------------------------------------
def assemble(K, Ke, LM_e):
    """
    K: 总体刚度矩阵 (neq x neq)，将被修改
    Ke: 单元刚度矩阵 (ndof_e x ndof_e)
    LM_e: 当前单元的自由度编号列表（长度 ndof_e）
    """
    ndof_e = len(LM_e)
    for a in range(ndof_e):
        i = LM_e[a]
        for b in range(ndof_e):
            j = LM_e[b]
            K[i, j] += Ke[a, b]

# ------------------------------------------------------------
# 缩减法处理位移边界条件并求解
# ------------------------------------------------------------
def solve_by_reduction(K, f, fixed_dofs, fixed_vals):
    """
    K: 总体刚度矩阵 (neq x neq)
    f: 节点力向量 (neq)
    fixed_dofs: 已知位移的自由度编号列表 (0-index)
    fixed_vals: 对应的位移值
    返回: d (总位移向量), r (约束反力向量)
    """
    neq = len(f)
    # 已知自由度集合
    fixed = np.array(fixed_dofs, dtype=int)
    free = np.setdiff1d(np.arange(neq), fixed)

    # 分块
    K_EE = K[np.ix_(fixed, fixed)]
    K_FF = K[np.ix_(free, free)]
    K_EF = K[np.ix_(fixed, free)]

    d_E = np.array(fixed_vals)
    f_F = f[free]

    # 求解自由度的位移
    d_F = np.linalg.solve(K_FF, f_F - K_EF.T @ d_E)

    # 重构完整位移向量
    d = np.zeros(neq)
    d[fixed] = d_E
    d[free] = d_F

    # 计算约束反力
    r = np.zeros(neq)
    r[fixed] = K_EE @ d_E + K_EF @ d_F
    return d, r

# ------------------------------------------------------------
# 后处理：计算单元应力及轴力
# ------------------------------------------------------------
def postprocess_elements(coords, IEN, E_list, A_list, d, ndof=2, k_list=None):
    """
    计算单元应力及轴力。
    如果提供了 k_list（EA/L），则直接用 k_list 计算轴力，忽略 E_list 和 A_list。
    否则使用 E_list 和 A_list 计算。
    """
    nel = len(IEN)
    results = []
    for e in range(nel):
        n1, n2 = IEN[e]
        x1, y1 = coords[n1]
        x2, y2 = coords[n2]
        de = np.zeros(ndof * 2)
        de[0] = d[ndof * n1 + 0]
        de[1] = d[ndof * n1 + 1]
        de[2] = d[ndof * n2 + 0]
        de[3] = d[ndof * n2 + 1]
        dx = x2 - x1
        dy = y2 - y1
        L = np.hypot(dx, dy)
        c = dx / L
        s = dy / L
        # 计算应变
        strain = (-c * de[0] - s * de[1] + c * de[2] + s * de[3]) / L

        if k_list is not None:
            # 直接使用 EA/L 计算轴力：N = k * 伸长量，伸长量 = strain * L
            N = k_list[e] * strain * L
            stress = 0.0  # 若无 E 和 A，应力无法唯一确定
        else:
            stress = E_list[e] * strain
            N = stress * A_list[e]
        results.append((L, c, s, stress, N))
    return results

# ------------------------------------------------------------
# 主求解函数：输入模型数据，输出结果
# ------------------------------------------------------------
def solve_truss(model):
    """
    model 字典包含：
        'coords'   : 节点坐标列表，[(x0,y0), (x1,y1), ...]
        'IEN'      : 单元连接表，[[n1,n2], ...] (全局节点号 0-index)
        'E'        : 单元弹性模量列表
        'A'        : 单元截面积列表
        'fixed_dofs' : 约束自由度编号 (0-index)
        'fixed_vals' : 对应约束值
        'force_dofs' : 载荷自由度编号 (0-index)
        'force_vals' : 对应载荷值
        'ndof'     : 每节点自由度数 (一般2)
    """
    ndof = model.get('ndof', 2)
    nnp = len(model['coords'])
    neq = nnp * ndof
    # 前处理：生成 LM
    IEN = np.array(model['IEN'])
    LM = generate_LM(IEN, ndof)   # shape (ndof*nen, nel)

    # 初始化总体刚度矩阵和力向量
    K = np.zeros((neq, neq))
    f = np.zeros(neq)

    # 施加载荷
    for dof, val in zip(model['force_dofs'], model['force_vals']):
        f[dof] = val

    # 单元分析并组装
    nel = len(IEN)
    for e in range(nel):
        n1, n2 = IEN[e]
        x1, y1 = model['coords'][n1]
        x2, y2 = model['coords'][n2]
        # 根据模型数据选择传入参数
        if 'k' in model:
            Ke, L, (c, s), k = element_stiffness_2d(x1, y1, x2, y2, k=model['k'][e])
        else:
            Ke, L, (c, s), k = element_stiffness_2d(x1, y1, x2, y2, E=model['E'][e], A=model['A'][e])
        # 获取当前单元的自由度编号 (4个)
        LM_e = LM[:, e].flatten()
        assemble(K, Ke, LM_e)

    # 求解
    d, r = solve_by_reduction(K, f, model['fixed_dofs'], model['fixed_vals'])

    # 后处理
    E_list = model.get('E')
    A_list = model.get('A')
    k_list = model.get('k')  # 新增：获取 k 列表
    # 调用后处理时传入 k_list
    elem_results = postprocess_elements(model['coords'], IEN, E_list, A_list, d, ndof, k_list=k_list)

    return K, d, r, elem_results

def read_json_model(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ndof = data['ndof']
    nnp = data['nnp']
    # 构建节点坐标列表 (0-index)
    coords = [(data['x'][i], data['y'][i]) for i in range(nnp)]
    # 单元连接：IEN 中节点号从1开始，转为0-index
    IEN = [[n-1 for n in elem] for elem in data['IEN']]
    # 材料参数
    E = data.get('E')
    A = data.get('CArea')
    # 边界条件：自由度编号从1开始，转为0-index
    fixed_dofs = [dof-1 for dof in data['fixed_dof']]
    fixed_vals = data['fixed_value']
    # 载荷
    force_dofs = [dof-1 for dof in data['force_dof']]
    force_vals = data['force_value']

    model = {
        'coords': coords,
        'IEN': IEN,
        'E': E,
        'A': A,
        'fixed_dofs': fixed_dofs,
        'fixed_vals': fixed_vals,
        'force_dofs': force_dofs,
        'force_vals': force_vals,
        'ndof': ndof
    }
    # 如果 JSON 中提供的是 EA/L 形式，则使用 'k' 键
    if 'k' in data:
        model['k'] = data['k']
    return model


# ------------------------------------------------------------
# 新增接口：供2.4作业获取缩减后的平衡方程组
# ------------------------------------------------------------
def get_reduced_system(model):
    """
    根据模型数据组装总体刚度矩阵和载荷向量，
    并应用位移边界条件，返回缩减后的方程。

    参数:
        model: dict，与 solve_truss 相同的模型数据
    返回:
        K_FF : 缩减后的刚度矩阵 (n_free x n_free)
        rhs  : 右端项向量 (n_free)
        free_dofs : 自由度的全局编号列表 (0-index)
        fixed_dofs: 固定自由度的全局编号列表 (0-index)
    """
    ndof = model.get('ndof', 2)
    nnp = len(model['coords'])
    neq = nnp * ndof
    IEN = np.array(model['IEN'])
    LM = generate_LM(IEN, ndof)

    K = np.zeros((neq, neq))
    f = np.zeros(neq)

    for dof, val in zip(model['force_dofs'], model['force_vals']):
        f[dof] = val

    nel = len(IEN)
    for e in range(nel):
        n1, n2 = IEN[e]
        x1, y1 = model['coords'][n1]
        x2, y2 = model['coords'][n2]
        if 'k' in model:
            Ke, _, _, _ = element_stiffness_2d(x1, y1, x2, y2, k=model['k'][e])
        else:
            Ke, _, _, _ = element_stiffness_2d(x1, y1, x2, y2, E=model['E'][e], A=model['A'][e])
        LM_e = LM[:, e].flatten()
        assemble(K, Ke, LM_e)

    fixed_dofs = model['fixed_dofs']
    fixed_vals = model['fixed_vals']
    fixed = np.array(fixed_dofs, dtype=int)
    free = np.setdiff1d(np.arange(neq), fixed)

    K_FF = K[np.ix_(free, free)]
    K_EF = K[np.ix_(fixed, free)]
    d_E = np.array(fixed_vals)
    f_F = f[free]

    rhs = f_F - K_EF.T @ d_E
    return K_FF, rhs, free, fixed


def export_reduced_system(model, json_file):
    """
    根据模型数据组装总体刚度矩阵和载荷向量，
    应用位移边界条件得到缩减后的方程，
    并将结果导出为 JSON 文件（供 2.4 作业使用）。

    参数:
        model: dict，模型数据
        json_file: str，输出 JSON 文件路径
    """
    import json

    ndof = model.get('ndof', 2)
    nnp = len(model['coords'])
    neq = nnp * ndof
    IEN = np.array(model['IEN'])
    LM = generate_LM(IEN, ndof)

    K = np.zeros((neq, neq))
    f = np.zeros(neq)

    for dof, val in zip(model['force_dofs'], model['force_vals']):
        f[dof] = val

    nel = len(IEN)
    for e in range(nel):
        n1, n2 = IEN[e]
        x1, y1 = model['coords'][n1]
        x2, y2 = model['coords'][n2]
        if 'k' in model:
            Ke, _, _, _ = element_stiffness_2d(x1, y1, x2, y2, k=model['k'][e])
        else:
            Ke, _, _, _ = element_stiffness_2d(x1, y1, x2, y2, E=model['E'][e], A=model['A'][e])
        LM_e = LM[:, e].flatten()
        assemble(K, Ke, LM_e)

    fixed_dofs = model['fixed_dofs']
    fixed_vals = model['fixed_vals']
    fixed = np.array(fixed_dofs, dtype=int)
    free = np.setdiff1d(np.arange(neq), fixed)

    K_FF = K[np.ix_(free, free)]
    K_EF = K[np.ix_(fixed, free)]
    d_E = np.array(fixed_vals)
    f_F = f[free]

    rhs = f_F - K_EF.T @ d_E

    # 准备导出数据（将所有 numpy 数组转换为列表）
    export_data = {
        "Title": "2.3 truss reduced equation",
        "source_homework": "2-3 Global Stiffness Equations",
        "K_FF": K_FF.tolist(),
        "rhs": rhs.tolist(),
        "free_dofs": free.tolist(),
        "fixed_dofs": fixed.tolist(),
        "full_K": K.tolist(),
        "full_force": f.tolist(),
        "known_displacement": fixed_vals,
        "coords": model['coords'],
        "IEN": model['IEN'],
        "E": model.get('E'),
        "CArea": model.get('A'),
        "ndof": model.get('ndof', 2)
    }

    if 'k' in model:
        export_data['k'] = model['k']  # 导出 EA/L 列表
    else:
        export_data['E'] = model.get('E')
        export_data['CArea'] = model.get('A')

    with open(json_file, 'w', encoding='utf-8') as f_out:
        json.dump(export_data, f_out, indent=2)

    print(f"缩减方程已导出到 {json_file}")
# ------------------------------------------------------------
# 辅助输出函数
# ------------------------------------------------------------
def print_matrix(name, mat, fmt="%12.6f"):
    print(f"\n{name} =")
    for row in mat:
        print(" ".join(fmt % x for x in row))

def print_vector(name, vec, fmt="%12.6f"):
    print(f"\n{name} =")
    print(" ".join(fmt % x for x in vec))

# ------------------------------------------------------------
# 算例1：一维两单元杆结构
# ------------------------------------------------------------
def example1():
    print("\n" + "=" * 60)
    print("算例1：一维两单元杆结构")
    print("=" * 60)

    # 节点坐标： 1:(0,0), 2:(1,0), 3:(2,0)
    coords = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    IEN = [[0, 1], [1, 2]]
    k_vals = [100.0, 200.0]

    # 边界条件：固定所有 y 方向自由度，以及节点1的 x 方向
    # 自由度编号: 节点1: x=0, y=1; 节点2: x=2, y=3; 节点3: x=4, y=5
    fixed_dofs = [0, 1, 3, 5]  # 节点1 x,y; 节点2 y; 节点3 y 固定
    fixed_vals = [0.0, 0.0, 0.0, 0.0]
    # 节点3 x方向力10
    force_dofs = [4]  # 节点3 x自由度
    force_vals = [10.0]

    model = {
        'coords': coords,
        'IEN': IEN,
        'k': k_vals,
        'fixed_dofs': fixed_dofs,
        'fixed_vals': fixed_vals,
        'force_dofs': force_dofs,
        'force_vals': force_vals,
        'ndof': 2
    }

    K, d, r, elem_results = solve_truss(model)

    # 输出总体刚度矩阵
    K_x = K[[0, 2, 4], :][:, [0, 2, 4]]
    print(f"\n1.总体刚度矩阵:")
    print_matrix("K_x", K_x, "%10.2f")

    # 提取 x 自由度位移
    u1 = d[0]
    u2 = d[2]
    u3 = d[4]
    print(f"\n2.节点位移: u1 = {u1:.6f}, u2 = {u2:.6f}, u3 = {u3:.6f}")

    # 反力：节点1 x方向反力
    r1 = r[0]
    print(f"\n3.节点1 x方向约束反力 = {r1:.6f} ")

    # 性质检查
    print("\n4.性质检查:")
    det_before = np.linalg.det(K_x)
    print(f"施加边界条件前刚度矩阵的行列式 = {det_before:.2e} -> {'奇异' if abs(det_before) < 1e-8 else '非奇异'}")
    # 缩减后刚度矩阵（对应 x 自由度 u2, u3）
    free_dofs_x = [1, 2]  # 对应 u2, u3 在 K_x 中的索引
    K_x_FF = K_x[np.ix_(free_dofs_x, free_dofs_x)]
    det_after = np.linalg.det(K_x_FF)
    print(
    f"施加边界条件后刚度矩阵的行列式 = {det_after:.2f} -> {'非奇异' if det_after > 1e-8 else '奇异'}")

    export_reduced_system(model, "算例1.json")

# ------------------------------------------------------------
# 算例2：二维两杆桁架结构
# ------------------------------------------------------------
def example2():
    print("\n" + "="*60)
    print("算例2：二维两杆桁架结构")
    print("="*60)

    # 节点坐标：1:(1,0), 2:(0,0), 3:(1,1)
    coords = [(1.0, 0.0), (0.0, 0.0), (1.0, 1.0)]
    # 单元连接：单元1: 1-3 , 单元2: 2-3 （0-index: 0-2, 1-2）
    IEN = [[0, 2], [1, 2]]
    E_vals = [1.0, 1.0]
    A_vals = [1.0, 1.0]
    # 边界条件：节点1固定 (自由度0,1), 节点2固定 (自由度2,3)
    fixed_dofs = [0, 1, 2, 3]
    fixed_vals = [0.0, 0.0, 0.0, 0.0]
    # 节点3载荷：Fx=10, Fy=0 (节点3索引2, 自由度4:x, 5:y)
    force_dofs = [4, 5]
    force_vals = [10.0, 0.0]

    model = {
        'coords': coords,
        'IEN': IEN,
        'E': E_vals,
        'A': A_vals,
        'fixed_dofs': fixed_dofs,
        'fixed_vals': fixed_vals,
        'force_dofs': force_dofs,
        'force_vals': force_vals,
        'ndof': 2
    }

    print("\n1.验证程序自动生成LM 矩阵:")
    ndof = model['ndof']
    IEN_arr = np.array(model['IEN'])
    LM = generate_LM(IEN_arr, ndof)  # 自动生成对号矩阵
    print("自动生成的 LM 矩阵:")
    print("单元1自由度编号: ", LM[:, 0])  # 应为 [0, 1, 4, 5]
    print("单元2自由度编号: ", LM[:, 1])  # 应为 [2, 3, 4, 5]

    # 理论值（节点编号从0开始，每个节点2个自由度）
    expected_LM = np.array([[0, 2],
                            [1, 3],
                            [4, 4],
                            [5, 5]], dtype=int)
    if np.array_equal(LM, expected_LM):
        print("LM 矩阵生成正确")
    else:
        print("LM 矩阵生成错误")

    K, d, r, elem_results = solve_truss(model)


    # 位移
    u3 = d[4]
    v3 = d[5]
    print(f"\n2.节点3位移: u3 = {u3:.6f}, v3 = {v3:.6f}")

    # 单元应力
    print("\n3.单元检查:")
    for e, (L, c, s, stress, N) in enumerate(elem_results):
        print(f"单元{e+1}: 长度={L:.4f}, 方向余弦(c,s)=({c:.6f},{s:.6f}), 应力={stress:.6f}, 轴力={N:.6f}")

    # 理论：单元1应力-10，单元2应力14.142136
    # 性质检查
    print("\n4.性质检查:")
    print("总体刚度矩阵对称性:", np.allclose(K, K.T))
    det_before = np.linalg.det(K)
    print(f"施加边界条件前刚度矩阵的行列式 = {det_before:.2e} -> {'奇异' if abs(det_before) < 1e-8 else '非奇异'}")
    # 缩减后刚度矩阵 (去掉约束自由度)
    free_dofs = [4,5]
    K_FF = K[np.ix_(free_dofs, free_dofs)]
    det_after = np.linalg.det(K_FF)
    print(
        f"施加边界条件后刚度矩阵的行列式 = {det_after:.2f} -> {'非奇异' if det_after > 1e-8 else '奇异'}")

    export_reduced_system(model, "算例2.json")

# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
if __name__ == "__main__":
    json_file = "D:/application/Git/FEM-BOOK/20260602/truss1.json"
    print(f"从 JSON 文件读取模型: {json_file}")
    model = read_json_model(json_file)
    K, d, r, elem_results = solve_truss(model)

    print("\n=== 求解结果 ===")
    print("节点位移:")
    nnp = len(model['coords'])
    ndof = model['ndof']
    for i in range(nnp):
        print(f"节点{i + 1}: u={d[ndof * i]:.6f}, v={d[ndof * i + 1]:.6f}")
    print("\n单元应力/轴力:")
    for e, (L, c, s, stress, N) in enumerate(elem_results):
        print(f"单元{e + 1}: 应力={stress:.6f}, 轴力={N:.6f}")

    else:
        # 无参数时运行原有算例
        example1()
        example2()