import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# ============================================================
# 设置字体
# ============================================================
plt.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 负号
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']  # 不使用中文字体


# ============================================================
# 1. 基本割圆术：直接计算 π_n = n * sin(π/n)
# ============================================================
def pi_basic(n):
    """基本割圆术：圆内接正 n 边形求 π 的近似值"""
    if n == 0:
        return 0.0
    return n * math.sin(math.pi / n)


# ============================================================
# 2. 生成边数序列和 π 近似序列
# ============================================================
def generate_sequences():
    """生成 n = 1, 2, 4, 8, 16, 32, 64, 128, 256 的 π 近似值"""
    n_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    pi_list = [pi_basic(n) for n in n_list]
    return n_list, pi_list


# ============================================================
# 3. 拟合斜率
# ============================================================
def fit_slope(h_values, errors, skip_first=2):
    """
    在对数坐标下拟合直线斜率

    参数:
        h_values: h = 1/n 数组
        errors: 误差数组
        skip_first: 跳过前几个不稳定点

    返回:
        slope: 拟合斜率（收敛阶）
    """
    min_len = min(len(h_values), len(errors))
    h_values = h_values[:min_len]
    errors = errors[:min_len]

    if skip_first < len(h_values):
        h_values = h_values[skip_first:]
        errors = errors[skip_first:]

    valid_idx = [i for i, e in enumerate(errors) if e > 1e-15 and h_values[i] > 0]
    if len(valid_idx) < 2:
        return 0.0

    log_h = np.log([h_values[i] for i in valid_idx])
    log_e = np.log([errors[i] for i in valid_idx])

    def linear(x, a, b):
        return a + b * x

    try:
        params, _ = curve_fit(linear, log_h, log_e)
        return params[1]
    except:
        return 0.0


# ============================================================
# 4. 主程序：生成绝对误差收敛图
# ============================================================
def plot_convergence():
    """生成基本割圆术的绝对误差收敛图"""

    print("=" * 70)
    print("基本割圆术求π - 绝对误差收敛图生成程序")
    print("=" * 70)

    # 生成数据
    print("\n正在生成边数序列: n = 1, 2, 4, 8, 16, 32, 64, 128, 256")
    n_list, pi_list = generate_sequences()

    print(f"边数列表: {n_list}")
    print(f"π近似值: {[f'{x:.10f}' for x in pi_list]}")

    # 真实 π 值
    pi_true = math.pi
    print(f"\nπ 的真值: {pi_true:.15f}")

    # 计算绝对误差
    h_list = [1.0 / n for n in n_list]
    error_list = [abs(pi_true - pi) for pi in pi_list]

    print(f"\n详细数据:")
    print(f"  n = {n_list}")
    print(f"  h = {[f'{x:.4f}' for x in h_list]}")
    print(f"  误差 = {[f'{x:.10e}' for x in error_list]}")

    # 拟合斜率
    slope = fit_slope(h_list, error_list, skip_first=2)

    print(f"\n拟合结果:")
    print(f"  基本割圆术拟合斜率: {slope:.2f}")

    # ============================================================
    # 绘图：绝对误差收敛图
    # ============================================================
    plt.figure(figsize=(10, 8))

    # 原始数据点（基本割圆术）
    plt.loglog(h_list, error_list, 'o-',
               label=f'Basic $\pi_n$ (slope = {slope:.2f})',
               linewidth=2, markersize=8, color='blue')

    plt.xlabel('h = 1/n (mesh size)', fontsize=14)
    plt.ylabel('Absolute Error |π - π_n|', fontsize=14)
    plt.title('Convergence of π Approximation (log-log scale)', fontsize=14, fontweight='bold')

    plt.grid(True, alpha=0.3, which='both', linestyle='--')
    plt.legend(loc='upper left', fontsize=12)

    # 显示拟合斜率值
    text_str = f'Fitted slope:\nBasic π_n: {slope:.2f}'

    min_error = min([e for e in error_list if e > 0])
    plt.text(0.003, min_error * 5, text_str, fontsize=11,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()

    # ============================================================
    # 保存图片
    # ============================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, 'pi_convergence_basic.png')

    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n图片已保存为: {save_path}")

    plt.show()

    # ============================================================
    # 打印详细数据表格
    # ============================================================
    print("\n" + "=" * 60)
    print(f"{'n':<8} {'π_n':<20} {'Absolute Error':<18} {'h = 1/n':<12}")
    print("=" * 60)

    for i in range(len(n_list)):
        print(f"{n_list[i]:<8} {pi_list[i]:<20.12f} {error_list[i]:<18.10e} {h_list[i]:<12.4f}")

    print("=" * 60)

    return slope


# ============================================================
# 5. 程序入口
# ============================================================
if __name__ == "__main__":
    slope = plot_convergence()

    print("\nProgram completed!")
    print(f"Convergence order of basic method: {slope:.2f}")