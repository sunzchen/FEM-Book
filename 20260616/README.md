\# 一维对流扩散方程有限元求解程序



\## 硬件环境

CPU:：13th Gen Intel(R) Core(TM) i7-13700F (2.10 GHz)

内存：16GB

操作系统：Microsoft Windows 11 专业版



\## 软件环境

Python：3.14.0

NumPy：2.4.4

Matplotlib：3.10.9





\## 并行配置

线程数：24



\## 运行方法

1.直接运行

python 3\_8Advection-Diffusion Equation.py



\##程序将自动依次执行：

1.基本任务（Pe=0.1 和 Pe=3.0）

对每个 Pe，分别使用三种格式求解

输出最大节点误差

绘制数值解与精确解的对比图（保存为 Pe\_0.1\_comparison.png 和 Pe\_3.0\_comparison.png）

2.矩阵性质分析（仅对 Pe=3.0）

输出标准 Galerkin 总体矩阵（施加边界条件前/后）的对称性、特征值范围和正定性判断

3.网格加密收敛性研究

固定物理参数（以 Pe=3.0, nel=20 为基准）

改变单元数 nel = \[10, 20, 40, 80]

输出误差汇总表

绘制误差收敛曲线（保存为 convergence\_study.png）







