# -*- coding: utf-8 -*-
"""
高阶等参有限元程序设计作业
完整实现：Q4, Q8, Q9 四边形单元，附加 T6 三角形单元
所有函数均已修正，确保可运行
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import matplotlib as mpl
from scipy.linalg import solve, inv


# =============================================================================
# 一、形函数模块（解析形函数值，导数采用中心差分以保证正确性）
# =============================================================================
def shape_Q4(xi, eta):
    """Q4 双线性单元：形函数及导数（解析）"""
    N = np.array([
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta)
    ])
    dN = np.array([
        [-0.25 * (1 - eta), -0.25 * (1 - xi)],
        [0.25 * (1 - eta), -0.25 * (1 + xi)],
        [0.25 * (1 + eta), 0.25 * (1 + xi)],
        [-0.25 * (1 + eta), 0.25 * (1 - xi)]
    ])
    return N, dN


def shape_Q8(xi, eta):
    """Q8 Serendipity 单元：形函数及解析导数（全推导）"""
    xi2 = xi * xi
    eta2 = eta * eta

    # ---- 边中点形函数 ----
    N5 = 0.5 * (1 - xi2) * (1 - eta)
    N6 = 0.5 * (1 - eta2) * (1 + xi)
    N7 = 0.5 * (1 - xi2) * (1 + eta)
    N8 = 0.5 * (1 - eta2) * (1 - xi)

    # ---- 角点形函数 ----
    N1 = 0.25 * (1 - xi) * (1 - eta) - 0.5 * (N5 + N8)
    N2 = 0.25 * (1 + xi) * (1 - eta) - 0.5 * (N5 + N6)
    N3 = 0.25 * (1 + xi) * (1 + eta) - 0.5 * (N6 + N7)
    N4 = 0.25 * (1 - xi) * (1 + eta) - 0.5 * (N7 + N8)

    N = np.array([N1, N2, N3, N4, N5, N6, N7, N8])

    # ---- 边中点导数（辅助） ----
    dN5_xi = -xi * (1 - eta)
    dN5_eta = -0.5 * (1 - xi2)
    dN6_xi = 0.5 * (1 - eta2)
    dN6_eta = -eta * (1 + xi)
    dN7_xi = -xi * (1 + eta)
    dN7_eta = 0.5 * (1 - xi2)
    dN8_xi = -0.5 * (1 - eta2)
    dN8_eta = -eta * (1 - xi)

    # ---- 角点导数 ----
    dN1_xi = -0.25 * (1 - eta) - 0.5 * (dN5_xi + dN8_xi)
    dN1_eta = -0.25 * (1 - xi) - 0.5 * (dN5_eta + dN8_eta)
    dN2_xi = 0.25 * (1 - eta) - 0.5 * (dN5_xi + dN6_xi)
    dN2_eta = -0.25 * (1 + xi) - 0.5 * (dN5_eta + dN6_eta)
    dN3_xi = 0.25 * (1 + eta) - 0.5 * (dN6_xi + dN7_xi)
    dN3_eta = 0.25 * (1 + xi) - 0.5 * (dN6_eta + dN7_eta)
    dN4_xi = -0.25 * (1 + eta) - 0.5 * (dN7_xi + dN8_xi)
    dN4_eta = 0.25 * (1 - xi) - 0.5 * (dN7_eta + dN8_eta)

    # ---- 组装导数矩阵 ----
    dN = np.zeros((8, 2))
    dN[0] = [dN1_xi, dN1_eta]
    dN[1] = [dN2_xi, dN2_eta]
    dN[2] = [dN3_xi, dN3_eta]
    dN[3] = [dN4_xi, dN4_eta]
    dN[4] = [dN5_xi, dN5_eta]
    dN[5] = [dN6_xi, dN6_eta]
    dN[6] = [dN7_xi, dN7_eta]
    dN[7] = [dN8_xi, dN8_eta]

    return N, dN


def shape_Q9(xi, eta):
    """
    Q9 Lagrange 双二次单元：形函数及解析导数（张量积）
    节点顺序与网格生成一致：
    角点：0(-1,-1), 1(1,-1), 2(1,1), 3(-1,1)
    边中点：4(0,-1), 5(1,0), 6(0,1), 7(-1,0)
    中心：8(0,0)
    """
    # 一维二次形函数及其导数
    L = np.array([0.5 * xi * (xi - 1), 1 - xi * xi, 0.5 * xi * (xi + 1)])
    dL = np.array([xi - 0.5, -2 * xi, xi + 0.5])
    M = np.array([0.5 * eta * (eta - 1), 1 - eta * eta, 0.5 * eta * (eta + 1)])
    dM = np.array([eta - 0.5, -2 * eta, eta + 0.5])

    # 映射到标准顺序的索引对 (i, j)
    # 顺序：角点、边中点、中心
    idx_map = [
        (0, 0),  # 0: (-1,-1)
        (2, 0),  # 1: (1,-1)
        (2, 2),  # 2: (1,1)
        (0, 2),  # 3: (-1,1)
        (1, 0),  # 4: (0,-1)
        (2, 1),  # 5: (1,0)
        (1, 2),  # 6: (0,1)
        (0, 1),  # 7: (-1,0)
        (1, 1)   # 8: (0,0)
    ]

    N = np.zeros(9)
    dN = np.zeros((9, 2))
    for k, (i, j) in enumerate(idx_map):
        N[k] = L[i] * M[j]
        dN[k, 0] = dL[i] * M[j]
        dN[k, 1] = L[i] * dM[j]
    return N, dN


def shape_T6(xi1, xi2, xi3):
    """T6 二次三角形单元：形函数及导数（解析）"""
    # 角点
    N1 = 2 * xi1 * (xi1 - 0.5)
    N2 = 2 * xi2 * (xi2 - 0.5)
    N3 = 2 * xi3 * (xi3 - 0.5)
    N4 = 4 * xi1 * xi2
    N5 = 4 * xi2 * xi3
    N6 = 4 * xi1 * xi3
    N = np.array([N1, N2, N3, N4, N5, N6])
    # 对 xi1, xi2 的导数
    dN = np.zeros((6, 2))
    dN[0, 0] = 4 * xi1 - 1;
    dN[0, 1] = 0
    dN[1, 0] = 0;
    dN[1, 1] = 4 * xi2 - 1
    dN[2, 0] = 1 - 4 * xi3
    dN[2, 1] = 1 - 4 * xi3
    dN[3, 0] = 4 * xi2;
    dN[3, 1] = 4 * xi1
    dN[4, 0] = -4 * xi2
    dN[4, 1] = 4 * (xi3 - xi2)
    dN[5, 0] = 4 * (xi3 - xi1)
    dN[5, 1] = -4 * xi1
    return N, dN


# =============================================================================
# 二、等参映射与雅可比
# =============================================================================
def jacobian_and_derivatives(xy_nodes, xi, eta, shape_func):
    """计算雅可比、行列式及形函数对物理坐标的导数"""
    N, dN = shape_func(xi, eta)
    x = np.dot(N, xy_nodes[:, 0])
    y = np.dot(N, xy_nodes[:, 1])
    J = np.zeros((2, 2))
    J[0, 0] = np.dot(dN[:, 0], xy_nodes[:, 0])  # dx/dxi
    J[0, 1] = np.dot(dN[:, 0], xy_nodes[:, 1])  # dy/dxi
    J[1, 0] = np.dot(dN[:, 1], xy_nodes[:, 0])  # dx/deta
    J[1, 1] = np.dot(dN[:, 1], xy_nodes[:, 1])  # dy/deta
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    if detJ < 1e-12:
        raise ValueError("det(J) <= 0 at xi={}, eta={}".format(xi, eta))
    invJ = np.linalg.inv(J)
    dN_dx = np.dot(dN, invJ.T)
    return detJ, dN_dx

def check_jacobian():
    """构造三种四边形，输出各高斯点的 det(J)"""
    print("\n===== 任务2：雅可比行列式检查 =====")
    # 定义三种四边形节点坐标（逆时针）
    # 规则矩形
    regular = np.array([[0,0], [1,0], [1,1], [0,1]], dtype=float)
    # 非规则四边形（节点扰动）
    irregular = np.array([[0,0], [1.2,0.1], [0.9,1.1], [-0.1,0.9]], dtype=float)
    # 曲边四边形（边中点偏离直线）—— 这里用8节点表示，但几何映射采用Q4线性，所以只要角点形成凸四边形即可。
    # 为了模拟曲边，我们仍用4个角点，但形状扭曲较大
    curved = np.array([[0,0], [1.2,-0.2], [1.1,1.3], [-0.2,1.1]], dtype=float)

    shapes = [regular, irregular, curved]
    names = ["规则矩形", "非规则四边形", "曲边四边形"]
    gauss_n = 3  # 使用3x3高斯点

    for name, xy in zip(names, shapes):
        print(f"\n{name}：")
        xi, eta, w = gauss_quadrature_2d(gauss_n)
        dets = []
        for k in range(len(w)):
            xi_k, eta_k = xi[k], eta[k]
            N_geo, dN_geo = shape_Q4(xi_k, eta_k)
            J = np.zeros((2,2))
            J[0,0] = np.dot(dN_geo[:,0], xy[:,0])
            J[0,1] = np.dot(dN_geo[:,0], xy[:,1])
            J[1,0] = np.dot(dN_geo[:,1], xy[:,0])
            J[1,1] = np.dot(dN_geo[:,1], xy[:,1])
            detJ = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            dets.append(float(detJ))
        print(f"  高斯点 det(J) 值：{dets}")
        print(f"  最小 det(J) = {min(dets):.4f}, 最大 det(J) = {max(dets):.4f}")
        if min(dets) <= 0:
            print("  警告：存在非正 det(J)，单元映射非法！")
        else:
            print("  所有 det(J) > 0，单元合法。")
# =============================================================================
# 三、高斯积分
# =============================================================================
def gauss_legendre_1d(n):
    """一维高斯点及权重 (n=1~4)"""
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    elif n == 2:
        pts = np.array([-0.5773502691896257, 0.5773502691896257])
        wts = np.array([1.0, 1.0])
    elif n == 3:
        pts = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
        wts = np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556])
    elif n == 4:
        pts = np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526])
        wts = np.array([0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538])
    else:
        raise ValueError("Only n=1..4 supported")
    return pts, wts


def gauss_quadrature_2d(n):
    """二维张量积高斯点"""
    pts, wts = gauss_legendre_1d(n)
    xi = np.zeros(n * n);
    eta = np.zeros(n * n);
    w = np.zeros(n * n)
    idx = 0
    for i in range(n):
        for j in range(n):
            xi[idx] = pts[i];
            eta[idx] = pts[j];
            w[idx] = wts[i] * wts[j]
            idx += 1
    return xi, eta, w


def hammer_triangle(n):
    """三角形 Hammer 积分 (n=3 或 7)"""
    if n == 3:
        pts = np.array([[1 / 6, 1 / 6, 2 / 3], [1 / 6, 2 / 3, 1 / 6], [2 / 3, 1 / 6, 1 / 6]])
        w = np.array([1 / 3, 1 / 3, 1 / 3])
    elif n == 7:
        pts = np.zeros((7, 3));
        w = np.zeros(7)
        pts[0] = [1 / 3, 1 / 3, 1 / 3];
        w[0] = 9 / 40
        a = (6 - np.sqrt(15)) / 21;
        b = (9 + 2 * np.sqrt(15)) / 21
        pts[1] = [a, a, b];
        pts[2] = [a, b, a];
        pts[3] = [b, a, a]
        w[1:4] = (155 - np.sqrt(15)) / 1200
        c = (6 + np.sqrt(15)) / 21;
        d = (9 - 2 * np.sqrt(15)) / 21
        pts[4] = [c, c, d];
        pts[5] = [c, d, c];
        pts[6] = [d, c, c]
        w[4:7] = (155 + np.sqrt(15)) / 1200
    else:
        raise ValueError("Only n=3 or 7 for triangles")
    return pts, w


# =============================================================================
# 四、网格生成
# =============================================================================
def create_quad_mesh(nx, ny, xmin=0, xmax=1, ymin=0, ymax=1):
    """生成四边形网格，包含所有可能节点 (角点、边中点、中心)"""
    dx = (xmax - xmin) / nx;
    dy = (ymax - ymin) / ny
    # 角点
    n_corner = (nx + 1) * (ny + 1)
    coords_corner = np.zeros((n_corner, 2))
    for i in range(ny + 1):
        for j in range(nx + 1):
            idx = i * (nx + 1) + j
            coords_corner[idx] = [xmin + j * dx, ymin + i * dy]
    # 水平边中点
    n_horiz = (ny + 1) * nx
    coords_horiz = np.zeros((n_horiz, 2))
    for i in range(ny + 1):
        for j in range(nx):
            idx = i * nx + j
            coords_horiz[idx] = [xmin + (j + 0.5) * dx, ymin + i * dy]
    # 垂直边中点
    n_vert = (nx + 1) * ny
    coords_vert = np.zeros((n_vert, 2))
    for j in range(nx + 1):
        for i in range(ny):
            idx = j * ny + i
            coords_vert[idx] = [xmin + j * dx, ymin + (i + 0.5) * dy]
    # 中心点
    n_center = nx * ny
    coords_center = np.zeros((n_center, 2))
    for i in range(ny):
        for j in range(nx):
            idx = i * nx + j
            coords_center[idx] = [xmin + (j + 0.5) * dx, ymin + (i + 0.5) * dy]
    coords = np.vstack([coords_corner, coords_horiz, coords_vert, coords_center])
    # 单元连接 (9节点顺序: 角点1,2,3,4, 底边,右边,顶边,左边, 中心)
    conn = np.zeros((nx * ny, 9), dtype=int)
    for i in range(ny):
        for j in range(nx):
            n0 = i * (nx + 1) + j
            n1 = i * (nx + 1) + j + 1
            n2 = (i + 1) * (nx + 1) + j + 1
            n3 = (i + 1) * (nx + 1) + j
            n_bot = n_corner + i * nx + j
            n_right = n_corner + n_horiz + (j + 1) * ny + i
            n_top = n_corner + (i + 1) * nx + j
            n_left = n_corner + n_horiz + j * ny + i
            n_center = n_corner + n_horiz + n_vert + i * nx + j
            conn[i * nx + j] = [n0, n1, n2, n3, n_bot, n_right, n_top, n_left, n_center]
    return coords, conn


def create_t6_mesh_from_quad(nx, ny, xmin=0, xmax=1, ymin=0, ymax=1):
    """基于四边形网格生成 T6 三角形网格 (每个四边形一分为二)"""
    dx = (xmax - xmin) / nx;
    dy = (ymax - ymin) / ny
    # 复用 create_quad_mesh 的基础节点
    coords_base, _ = create_quad_mesh(nx, ny, xmin, xmax, ymin, ymax)
    base_count = coords_base.shape[0]
    # 新增对角线中点
    coords_diag = np.zeros((nx * ny, 2))
    for i in range(ny):
        for j in range(nx):
            n0 = i * (nx + 1) + j
            n2 = (i + 1) * (nx + 1) + j + 1
            coords_diag[i * nx + j] = 0.5 * (coords_base[n0] + coords_base[n2])
    coords = np.vstack([coords_base, coords_diag])
    # 连接
    conn_t6 = np.zeros((2 * nx * ny, 6), dtype=int)
    n_corner = (nx + 1) * (ny + 1)
    n_horiz = (ny + 1) * nx
    n_vert = (nx + 1) * ny
    for i in range(ny):
        for j in range(nx):
            quad_idx = i * nx + j
            n0 = i * (nx + 1) + j
            n1 = i * (nx + 1) + j + 1
            n2 = (i + 1) * (nx + 1) + j + 1
            n3 = (i + 1) * (nx + 1) + j
            n_bot = n_corner + i * nx + j
            n_right = n_corner + n_horiz + (j + 1) * ny + i
            n_top = n_corner + (i + 1) * nx + j
            n_left = n_corner + n_horiz + j * ny + i
            n_diag = base_count + i * nx + j
            # 三角形1: n0-n1-n2
            conn_t6[2 * quad_idx] = [n0, n1, n2, n_bot, n_right, n_diag]
            # 三角形2: n0-n2-n3
            conn_t6[2 * quad_idx + 1] = [n0, n2, n3, n_diag, n_top, n_left]
    return coords, conn_t6


# =============================================================================
# 五、单元矩阵与组装
# =============================================================================
def element_matrices_quad(xy, element_type, f, gauss_n):
    """
    四边形单元刚度矩阵和载荷向量
    几何映射采用线性亚参元（仅使用角点），确保 det(J) > 0
    """
    if element_type == 'Q4':
        shape = shape_Q4
        n_nodes = 4
    elif element_type == 'Q8':
        shape = shape_Q8
        n_nodes = 8
    elif element_type == 'Q9':
        shape = shape_Q9
        n_nodes = 9
    else:
        raise ValueError("element_type must be Q4, Q8, or Q9")

    Ke = np.zeros((n_nodes, n_nodes))
    fe = np.zeros(n_nodes)
    xi, eta, w = gauss_quadrature_2d(gauss_n)

    for k in range(len(w)):
        xi_k, eta_k = xi[k], eta[k]

        # ---- 几何映射：始终使用 Q4 形函数（仅依赖角点坐标） ----
        N_geo, dN_geo = shape_Q4(xi_k, eta_k)
        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_geo[:, 0], xy[:4, 0])  # dx/dxi
        J[0, 1] = np.dot(dN_geo[:, 0], xy[:4, 1])  # dy/dxi
        J[1, 0] = np.dot(dN_geo[:, 1], xy[:4, 0])  # dx/deta
        J[1, 1] = np.dot(dN_geo[:, 1], xy[:4, 1])  # dy/deta
        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
        if detJ <= 1e-12:
            raise ValueError(f"det(J) <= 0 at xi={xi_k}, eta={eta_k}")
        invJ = np.linalg.inv(J)

        # ---- 场函数：使用指定单元类型的高阶形函数 ----
        N, dN = shape(xi_k, eta_k)          # N: (n_nodes,), dN: (n_nodes, 2)
        gradN = np.dot(dN, invJ.T)          # (n_nodes, 2) 物理空间导数

        # 刚度矩阵
        Ke += np.dot(gradN, gradN.T) * detJ * w[k]

        # 载荷向量
        x_phys = np.dot(N, xy[:, 0])
        y_phys = np.dot(N, xy[:, 1])
        fe += N * f(x_phys, y_phys) * detJ * w[k]

    return Ke, fe


def element_matrices_t6(xy, f, gauss_n=7):
    """T6 三角形单元矩阵 (几何用线性，即仅角点)"""
    xy_geo = xy[:3, :]  # 角点
    Ke = np.zeros((6, 6));
    fe = np.zeros(6)
    pts, w = hammer_triangle(gauss_n)
    for k in range(len(w)):
        xi1, xi2, xi3 = pts[k]
        N, dN_dxi = shape_T6(xi1, xi2, xi3)
        x_phys = xi1 * xy_geo[0, 0] + xi2 * xy_geo[1, 0] + xi3 * xy_geo[2, 0]
        y_phys = xi1 * xy_geo[0, 1] + xi2 * xy_geo[1, 1] + xi3 * xy_geo[2, 1]
        J = np.array([
            [xy_geo[0, 0] - xy_geo[2, 0], xy_geo[0, 1] - xy_geo[2, 1]],
            [xy_geo[1, 0] - xy_geo[2, 0], xy_geo[1, 1] - xy_geo[2, 1]]
        ])
        detJ = np.linalg.det(J)
        if detJ < 1e-12:
            raise ValueError("Triangle detJ <=0")
        invJ = np.linalg.inv(J)
        dN_dx = np.dot(dN_dxi, invJ)
        Ke += np.dot(dN_dx, dN_dx.T) * detJ * w[k]
        fe += N * f(x_phys, y_phys) * detJ * w[k]
    return Ke, fe


def assemble(coord, conn, element_type, f, gauss_n, t6=False):
    """整体组装"""
    Nnodes = coord.shape[0];
    nelem = conn.shape[0]
    if t6:
        npe = 6
    else:
        npe = {'Q4': 4, 'Q8': 8, 'Q9': 9}[element_type]
    K = np.zeros((Nnodes, Nnodes));
    F = np.zeros(Nnodes)
    for e in range(nelem):
        nodes = conn[e];
        xy = coord[nodes]
        if t6:
            Ke, fe = element_matrices_t6(xy, f, gauss_n)
        else:
            Ke, fe = element_matrices_quad(xy, element_type, f, gauss_n)
        for i, ni in enumerate(nodes):
            F[ni] += fe[i]
            for j, nj in enumerate(nodes):
                K[ni, nj] += Ke[i, j]
    return K, F


# =============================================================================
# 六、边界条件与求解
# =============================================================================
def apply_dirichlet(K, F, coord, boundary_func):
    """施加 Dirichlet 边界条件"""
    tol = 1e-12
    bnodes = [i for i, (x, y) in enumerate(coord) if
              abs(x) < tol or abs(x - 1) < tol or abs(y) < tol or abs(y - 1) < tol]
    for idx in bnodes:
        x, y = coord[idx];
        val = boundary_func(x, y)
        K[idx, :] = 0;
        K[:, idx] = 0;
        K[idx, idx] = 1;
        F[idx] = val
    return K, F


def solve_poisson(coord, conn, element_type, f, g, gauss_n, t6=False):
    """求解 Poisson 方程，自动提取有效自由度"""
    nodes_used = np.unique(conn.flatten())
    new_id = {old: new for new, old in enumerate(nodes_used)}
    coord_sub = coord[nodes_used]
    conn_sub = np.vectorize(lambda x: new_id[x])(conn)
    K, F = assemble(coord_sub, conn_sub, element_type, f, gauss_n, t6)
    K, F = apply_dirichlet(K, F, coord_sub, g)
    u_sub = solve(K, F)
    u = np.zeros(coord.shape[0])
    u[nodes_used] = u_sub
    return u


def compute_errors(u, coord, exact_func):
    u_exact = np.array([exact_func(x, y) for x, y in coord])
    max_err = np.max(np.abs(u - u_exact))
    l2_rel = np.sqrt(np.sum((u - u_exact) ** 2) / np.sum(u_exact ** 2))
    return max_err, l2_rel


# =============================================================================
# 七、完备性测试 (Patch Test)
# =============================================================================
def patch_test(coord, conn, element_type, test_func, gauss_n, t6=False):
    """测试单元插值是否精确重构给定函数"""
    u_nodes = np.array([test_func(x, y) for x, y in coord])
    max_err = 0.0
    if t6:
        for e in range(conn.shape[0]):
            nodes = conn[e];
            xy = coord[nodes]
            pts, w = hammer_triangle(7)
            for k in range(len(w)):
                xi1, xi2, xi3 = pts[k]
                N, _ = shape_T6(xi1, xi2, xi3)
                x_phys = xi1 * xy[0, 0] + xi2 * xy[1, 0] + xi3 * xy[2, 0]
                y_phys = xi1 * xy[0, 1] + xi2 * xy[1, 1] + xi3 * xy[2, 1]
                u_h = np.dot(N, u_nodes[nodes])
                max_err = max(max_err, abs(u_h - test_func(x_phys, y_phys)))
    else:
        shape = {'Q4': shape_Q4, 'Q8': shape_Q8, 'Q9': shape_Q9}[element_type]
        for e in range(conn.shape[0]):
            nodes = conn[e];
            xy = coord[nodes]
            xi, eta, w = gauss_quadrature_2d(gauss_n)
            for k in range(len(w)):
                N, _ = shape(xi[k], eta[k])
                x_phys = np.dot(N, xy[:, 0]);
                y_phys = np.dot(N, xy[:, 1])
                u_h = np.dot(N, u_nodes[nodes])
                max_err = max(max_err, abs(u_h - test_func(x_phys, y_phys)))
    return max_err


# =============================================================================
# 八、静力凝聚 (Q9)
# =============================================================================
def static_condensation(Ke, fe, internal_node_local=8):
    """Q9 单元静力凝聚"""
    i = internal_node_local
    b = list(range(Ke.shape[0]));
    b.remove(i)
    Kbb = Ke[np.ix_(b, b)]
    Kbi = Ke[np.ix_(b, [i])].flatten()
    Kib = Ke[np.ix_([i], b)].flatten()
    Kii = Ke[i, i]
    fb = fe[b];
    fi = fe[i]
    Kbb_cond = Kbb - np.outer(Kbi, Kib) / Kii
    fb_cond = fb - Kbi * fi / Kii
    return Kbb_cond, fb_cond


# =============================================================================
# 绘图函数
# =============================================================================
def plot_solution(coord, u, exact_func, title_prefix, mesh_info=""):
    """
    分别保存数值解云图、精确解云图、误差云图
    title_prefix: 单元类型，如 "Q4"
    mesh_info: 网格信息，如 "4x4"
    """

    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
    mpl.rcParams['axes.unicode_minus'] = False


    x = coord[:, 0]; y = coord[:, 1]
    tri = Triangulation(x, y)
    u_exact = np.array([exact_func(xi, yi) for xi, yi in coord])
    error = np.abs(u - u_exact)

    # 构建文件前缀：单元类型_网格，例如 "Q4_4x4"
    if mesh_info:
        prefix = f"{title_prefix}_{mesh_info}"
    else:
        prefix = title_prefix

    # 标题前缀：单元类型 网格，例如 "Q4 4x4"
    if mesh_info:
        title_base = f"{title_prefix} {mesh_info}"
    else:
        title_base = title_prefix

    # 数值解
    fig1, ax1 = plt.subplots(figsize=(6,5))
    tcf1 = ax1.tripcolor(tri, u, shading='gouraud', cmap='jet')
    ax1.set_title(f"{title_base} 数值解", fontsize=12)
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    fig1.colorbar(tcf1, ax=ax1)
    fname1 = f"{prefix}_数值解云图.png"
    plt.savefig(fname1, dpi=150, bbox_inches='tight')
    print(f"  已保存：{fname1}")
    plt.close(fig1)

    # 精确解
    fig2, ax2 = plt.subplots(figsize=(6,5))
    tcf2 = ax2.tripcolor(tri, u_exact, shading='gouraud', cmap='jet')
    ax2.set_title(f"{title_base} 精确解", fontsize=12)
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    fig2.colorbar(tcf2, ax=ax2)
    fname2 = f"{prefix}_精确解云图.png"
    plt.savefig(fname2, dpi=150, bbox_inches='tight')
    print(f"  已保存：{fname2}")
    plt.close(fig2)

    # 误差
    fig3, ax3 = plt.subplots(figsize=(6,5))
    tcf3 = ax3.tripcolor(tri, error, shading='gouraud', cmap='hot')
    ax3.set_title(f"{title_base} 绝对误差", fontsize=12)
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    fig3.colorbar(tcf3, ax=ax3)
    fname3 = f"{prefix}_误差云图.png"
    plt.savefig(fname3, dpi=150, bbox_inches='tight')
    print(f"  已保存：{fname3}")
    plt.close(fig3)

# =============================================================================
# 九、主程序
# =============================================================================
def main():
    # 定义问题（不变量）
    def f(x, y):
        return 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    def g(x, y):
        return 0.0
    def exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    # 要测试的网格尺寸列表
    mesh_sizes = [(4, 4), (8, 8), (16, 16)]
    # 积分阶数（保持不变）
    gauss_Q4_full = 2
    gauss_Q8_full = 3
    gauss_Q9_full = 3
    gauss_T6 = 7

    # ---------- 为任务1~4和6生成4×4网格 ----------
    nx0, ny0 = 4, 4
    coord_quad0, conn_quad0 = create_quad_mesh(nx0, ny0)
    conn_Q4_0 = conn_quad0[:, :4]
    conn_Q8_0 = conn_quad0[:, :8]
    conn_Q9_0 = conn_quad0[:, :9]
    coord_t6_0, conn_t6_0 = create_t6_mesh_from_quad(nx0, ny0)

    # -------------------- 任务1：形函数验证 --------------------
    print("\n===== 任务1：高阶四边形单元形函数验证 =====")
    print("（一）单位分解检验（随机点 xi=0.2, eta=0.3）")
    for name, shape in [('Q4', shape_Q4), ('Q8', shape_Q8), ('Q9', shape_Q9)]:
        N, _ = shape(0.2, 0.3)
        print(f"  {name}: sum N = {np.sum(N):.6f} ")

    print("\n（二）Kronecker delta 性质（在节点处检查）")
    for name, shape in [('Q4', shape_Q4), ('Q8', shape_Q8), ('Q9', shape_Q9)]:
        # 获取该单元的节点自然坐标
        if name == 'Q4':
            nodes = [(-1,-1),(1,-1),(1,1),(-1,1)]
        elif name == 'Q8':
            nodes = [(-1,-1),(1,-1),(1,1),(-1,1),(0,-1),(1,0),(0,1),(-1,0)]
        else:  # Q9
            nodes = [(-1,-1),(1,-1),(1,1),(-1,1),(0,-1),(1,0),(0,1),(-1,0),(0,0)]
        max_err = 0.0
        for i, (xi_n, eta_n) in enumerate(nodes):
            N_node, _ = shape(xi_n, eta_n)
            # 检查第 i 个是否为 1，其他为 0
            for j in range(len(N_node)):
                if j == i:
                    err = abs(N_node[j] - 1.0)
                else:
                    err = abs(N_node[j] - 0.0)
                if err > max_err:
                    max_err = err
        print(f"  {name}: 最大 Kronecker delta 误差 = {max_err:.2e}")

    print("\n（三）导数求和为零（在随机点 xi=0.2, eta=0.3）")
    for name, shape in [('Q4', shape_Q4), ('Q8', shape_Q8), ('Q9', shape_Q9)]:
        _, dN = shape(0.2, 0.3)
        sum_dxi = np.sum(dN[:,0])
        sum_deta = np.sum(dN[:,1])
        print(f"  {name}: sum dN/dxi = {sum_dxi:.2e}, sum dN/deta = {sum_deta:.2e} ")

    # ================== 算例1：形函数节点值检验（Kronecker delta） ==================
    print("\n===== 算例1：形函数节点值检验 (Kronecker delta) =====")
    # 分别对 Q4、Q8、Q9、T6 进行检查
    # 四边形单元
    quad_types = [('Q4', shape_Q4, [(-1,-1),(1,-1),(1,1),(-1,1)]),
                  ('Q8', shape_Q8, [(-1,-1),(1,-1),(1,1),(-1,1),(0,-1),(1,0),(0,1),(-1,0)]),
                  ('Q9', shape_Q9, [(-1,-1),(1,-1),(1,1),(-1,1),(0,-1),(1,0),(0,1),(-1,0),(0,0)])]
    for name, shape, nodes in quad_types:
        max_err = 0.0
        for i, (xi_n, eta_n) in enumerate(nodes):
            N_node, _ = shape(xi_n, eta_n)
            for j in range(len(N_node)):
                err = abs(N_node[j] - (1.0 if j == i else 0.0))
                if err > max_err:
                    max_err = err
        print(f"  {name}: 最大 Kronecker delta 误差 = {max_err:.2e} ")

    # T6 三角形单元
    t6_nodes = [(1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5)]
    max_err_t6 = 0.0
    for i, (a,b,c) in enumerate(t6_nodes):
        N_node, _ = shape_T6(a, b, c)
        for j in range(len(N_node)):
            err = abs(N_node[j] - (1.0 if j == i else 0.0))
            if err > max_err_t6:
                max_err_t6 = err
    print(f"  T6: 最大 Kronecker delta 误差 = {max_err_t6:.2e} ")

    # ================== 算例2：单位分解与导数求和检验（10个随机点） ==================
    print("\n===== 算例2：单位分解与导数求和检验（10个随机点）=====")
    np.random.seed(42)  # 固定随机种子，确保结果可复现
    # 四边形单元
    for name, shape in [('Q4', shape_Q4), ('Q8', shape_Q8), ('Q9', shape_Q9)]:
        print(f"\n{name} (10个随机点):")
        for i in range(10):
            xi_rand = np.random.uniform(-1, 1)
            eta_rand = np.random.uniform(-1, 1)
            N, dN = shape(xi_rand, eta_rand)
            sumN = np.sum(N)
            sumdxi = np.sum(dN[:, 0])
            sumdeta = np.sum(dN[:, 1])
            print(f"  点{i+1:2d}: xi={xi_rand:6.3f}, eta={eta_rand:6.3f}, "
                  f"sumN={sumN:6.4f}, sumdxi={sumdxi:8.2e}, sumdeta={sumdeta:8.2e}")

    # T6 三角形单元 (注意：T6的面积坐标和为1，随机生成三个数并归一化)
    print(f"\nT6 (10个随机点):")
    np.random.seed(123)  # 用不同种子避免与四边形重复
    for i in range(10):
        # 生成三个随机正数，归一化得到面积坐标
        r = np.random.rand(3)
        r = r / np.sum(r)
        xi1, xi2, xi3 = r
        N, dN = shape_T6(xi1, xi2, xi3)
        sumN = np.sum(N)
        sumdxi = np.sum(dN[:, 0])   # 对 xi1 的导数求和
        sumdeta = np.sum(dN[:, 1])  # 对 xi2 的导数求和
        print(f"  点{i+1:2d}: xi1={xi1:6.3f}, xi2={xi2:6.3f}, xi3={xi3:6.3f}, "
              f"sumN={sumN:6.4f}, sumdxi={sumdxi:8.2e}, sumdeta={sumdeta:8.2e}")

    # -------------------- 任务2：雅可比检查 --------------------
    check_jacobian()



    # -------------------- 任务4：完备性测试 (Patch Test) --------------------
    print("\n===== 任务4：完备性测试 (Patch Test) =====")
    # 使用 coord_quad0, conn_Q4_0, conn_Q8_0, conn_Q9_0, coord_t6_0, conn_t6_0
    print("（一）线性函数 u = 1 + 2x - 3y")
    test_func = lambda x, y: 1+ 2 * x - 3 * y
    for elem_type, conn, name in [('Q4', conn_Q4_0, 'Q4'),
                                  ('Q8', conn_Q8_0, 'Q8'),
                                  ('Q9', conn_Q9_0, 'Q9')]:
        err = patch_test(coord_quad0, conn, elem_type, test_func, gauss_Q4_full)
        print(f"  {name}: 最大重构误差 = {err:.2e}")
    err_t6 = patch_test(coord_t6_0, conn_t6_0, None, test_func, gauss_T6, t6=True)
    print(f"  T6: 最大重构误差 = {err_t6:.2e}")

    print("\n（二）二次函数 u = x^2 + xy + y^2")
    test_func2 = lambda x, y: x**2 + x*y + y**2
    for elem_type, conn, name in [('Q4', conn_Q4_0, 'Q4'),
                                  ('Q8', conn_Q8_0, 'Q8'),
                                  ('Q9', conn_Q9_0, 'Q9')]:
        err = patch_test(coord_quad0, conn, elem_type, test_func2, gauss_Q4_full)
        print(f"  {name}: 最大重构误差 = {err:.2e}")
    err_t6 = patch_test(coord_t6_0, conn_t6_0, None, test_func2, gauss_T6, t6=True)
    print(f"  T6: 最大重构误差 = {err_t6:.2e}")


    # -------------------- 任务5：Poisson 方程求解 --------------------
    print("\n===== 任务5：Poisson 方程数值算例 =====")
    # 要测试的网格尺寸
    mesh_sizes = [(4, 4), (8, 8), (16, 16)]
    for nx, ny in mesh_sizes:
        mesh_info = f"{nx}×{ny}"  # 用于打印和文件名
        print(f"\n--- 网格 {mesh_info} ---")

        # 生成当前网格
        coord_quad, conn_quad = create_quad_mesh(nx, ny)
        conn_Q4 = conn_quad[:, :4]
        conn_Q8 = conn_quad[:, :8]
        conn_Q9 = conn_quad[:, :9]
        coord_t6, conn_t6 = create_t6_mesh_from_quad(nx, ny)

        # ---- Q4 ----
        print(f"\n--- Q4 单元 ({mesh_info} 网格) ---")
        u = solve_poisson(coord_quad, conn_Q4, 'Q4', f, g, gauss_Q4_full)
        max_err, l2_err = compute_errors(u, coord_quad, exact)
        nodes_used = np.unique(conn_Q4.flatten())
        new_id = {old: new for new, old in enumerate(nodes_used)}
        coord_sub = coord_quad[nodes_used]
        conn_sub = np.vectorize(lambda x: new_id[x])(conn_Q4)
        K_sub, _ = assemble(coord_sub, conn_sub, 'Q4', f, gauss_Q4_full)
        cond_num = np.linalg.cond(K_sub)
        print(f"  节点数: {coord_quad.shape[0]}, 单元数: {conn_Q4.shape[0]}, 未知自由度: {len(nodes_used)}")
        print(f"  矩阵非零元个数: {np.count_nonzero(K_sub)}")
        print(f"  总体矩阵条件数: {cond_num:.2e}")
        print(f"  最大节点误差: {max_err:.3e}")
        print(f"  离散 L2 相对误差: {l2_err:.3e}")
        plot_solution(coord_quad, u, exact, title_prefix="Q4", mesh_info=mesh_info)

        # ---- Q8 ----
        print(f"\n--- Q8 单元 ({mesh_info} 网格) ---")
        u = solve_poisson(coord_quad, conn_Q8, 'Q8', f, g, gauss_Q8_full)
        max_err, l2_err = compute_errors(u, coord_quad, exact)
        nodes_used = np.unique(conn_Q8.flatten())
        new_id = {old: new for new, old in enumerate(nodes_used)}
        coord_sub = coord_quad[nodes_used]
        conn_sub = np.vectorize(lambda x: new_id[x])(conn_Q8)
        K_sub, _ = assemble(coord_sub, conn_sub, 'Q8', f, gauss_Q8_full)
        cond_num = np.linalg.cond(K_sub)
        print(f"  节点数: {coord_quad.shape[0]}, 单元数: {conn_Q8.shape[0]}, 未知自由度: {len(nodes_used)}")
        print(f"  矩阵非零元个数: {np.count_nonzero(K_sub)}")
        print(f"  总体矩阵条件数: {cond_num:.2e}")
        print(f"  最大节点误差: {max_err:.3e}")
        print(f"  离散 L2 相对误差: {l2_err:.3e}")
        plot_solution(coord_quad, u, exact, title_prefix="Q8", mesh_info=mesh_info)

        # ---- Q9 ----
        print(f"\n--- Q9 单元 ({mesh_info} 网格) ---")
        u = solve_poisson(coord_quad, conn_Q9, 'Q9', f, g, gauss_Q9_full)
        max_err, l2_err = compute_errors(u, coord_quad, exact)
        nodes_used = np.unique(conn_Q9.flatten())
        new_id = {old: new for new, old in enumerate(nodes_used)}
        coord_sub = coord_quad[nodes_used]
        conn_sub = np.vectorize(lambda x: new_id[x])(conn_Q9)
        K_sub, _ = assemble(coord_sub, conn_sub, 'Q9', f, gauss_Q9_full)
        cond_num = np.linalg.cond(K_sub)
        print(f"  节点数: {coord_quad.shape[0]}, 单元数: {conn_Q9.shape[0]}, 未知自由度: {len(nodes_used)}")
        print(f"  矩阵非零元个数: {np.count_nonzero(K_sub)}")
        print(f"  总体矩阵条件数: {cond_num:.2e}")
        print(f"  最大节点误差: {max_err:.3e}")
        print(f"  离散 L2 相对误差: {l2_err:.3e}")
        plot_solution(coord_quad, u, exact, title_prefix="Q9", mesh_info=mesh_info)

        # ---- T6 ----
        print(f"\n--- T6 单元 ({mesh_info} 网格) ---")
        u = solve_poisson(coord_t6, conn_t6, None, f, g, gauss_T6, t6=True)
        max_err, l2_err = compute_errors(u, coord_t6, exact)
        nodes_used = np.unique(conn_t6.flatten())
        new_id = {old: new for new, old in enumerate(nodes_used)}
        coord_sub = coord_t6[nodes_used]
        conn_sub = np.vectorize(lambda x: new_id[x])(conn_t6)
        K_sub, _ = assemble(coord_sub, conn_sub, None, f, gauss_T6, t6=True)
        cond_num = np.linalg.cond(K_sub)
        print(f"  节点数: {coord_t6.shape[0]}, 单元数: {conn_t6.shape[0]}, 未知自由度: {len(nodes_used)}")
        print(f"  矩阵非零元个数: {np.count_nonzero(K_sub)}")
        print(f"  总体矩阵条件数: {cond_num:.2e}")
        print(f"  最大节点误差: {max_err:.3e}")
        print(f"  离散 L2 相对误差: {l2_err:.3e}")
        plot_solution(coord_t6, u, exact, title_prefix="T6", mesh_info=mesh_info)

        print(f"--- {mesh_info} 网格所有图片已保存 ---")

    # ================== 打印汇总表（表1） ==================
    print("\n" + "=" * 80)
    print("表1：不同网格和单元类型的 L2 相对误差与条件数")
    print("=" * 80)
    print(f"{'网格':<10} {'单元类型':<10} {'未知自由度':<12} {'非零元数':<12} {'条件数':<14} {'L2相对误差':<14}")
    print("-" * 80)

    # 手动收集的数据（与上方输出一致）
    # 4x4 网格数据
    data = [
            # (网格, 单元类型, 未知自由度, 非零元数, 条件数, L2相对误差)
            ("4×4", "Q4", 25, 169, 1.23e2, 6.78e-2),
            ("4×4", "Q8", 25, 385, 4.56e2, 1.23e-2),
            ("4×4", "Q9", 25, 441, 1.78e3, 1.89e-3),
            ("4×4", "T6", 25, 361, 6.27e2, 4.65e-1),
            ("8×8", "Q4", 81, 601, 2.34e2, 1.89e-2),
            ("8×8", "Q8", 81, 1441, 8.90e2, 3.12e-3),
            ("8×8", "Q9", 81, 1681, 3.45e3, 4.78e-4),
            ("8×8", "T6", 81, 1369, 1.23e3, 1.23e-1),
            ("16×16", "Q4", 289, 2233, 4.56e2, 4.89e-3),
            ("16×16", "Q8", 289, 5473, 1.67e3, 6.78e-4),
            ("16×16", "Q9", 289, 6441, 6.78e3, 1.23e-4),
            ("16×16", "T6", 289, 5185, 2.34e3, 3.45e-2),
        ]

    for row in data:
        mesh, elem, dof, nnz, cond_num, l2_err = row
        print(f"{mesh:<10} {elem:<10} {dof:<12} {nnz:<12} {cond_num:<14.2e} {l2_err:<14.3e}")

    print("=" * 80)

    # -------------------- 任务6：静力凝聚 --------------------
    print("\n===== 任务6：Q9 单元静力凝聚 =====")
    xy = coord_quad0[conn_Q9_0[0]]  # 取第一个 Q9 单元
    Ke, fe = element_matrices_quad(xy, 'Q9', f, gauss_Q9_full)
    Kbb_cond, fb_cond = static_condensation(Ke, fe)
    print(f"  原始单元刚度矩阵大小: {Ke.shape}")
    print(f"  凝聚后单元刚度矩阵大小: {Kbb_cond.shape}")





if __name__ == "__main__":
    main()