import matplotlib.pyplot as plt

# 提取自日志中的迭代数据 (1-20次)
iterations = list(range(1, 21))

# obj_primal 数据
obj_primal = [
    4.519866, 1.063478, 0.889972, 0.709489, 0.533262, 
    0.503635, 0.441065, 0.514157, 0.498884, 0.486394,
    0.556213, 0.552033, 0.580650, 0.578803, 0.547828,
    0.598431, 0.642436, 0.675019, 0.678053, 0.530535
]

# obj_dual 数据
obj_dual = [
    55.801272, 41.990481, 41.587251, 34.523462, 31.750819,
    30.344314, 30.281130, 29.893349, 28.549665, 26.050911,
    27.919573, 25.768731, 25.012382, 24.642579, 24.044456,
    26.570500, 23.067103, 22.337255, 21.900406, 21.044017
]

# obj_opt 数据
obj_opt = [
    23.576304, 5.567025, 5.849926, 7.606182, 3.471886,
    4.495634, 3.201272, 4.103510, 3.290904, 0.974854,
    3.640865, 3.552706, 1.292538, 1.780447, 2.684540,
    6.220471, 0.919677, 1.171262, 1.227603, 0.765143
]

# 设置绘图风格
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-paper') # 或使用 'ggplot'

# 绘制三条曲线
plt.plot(iterations, obj_primal, marker='o', linestyle='-', linewidth=2, label='Obj Primal (Primal Infeasibility)')
plt.plot(iterations, obj_dual, marker='s', linestyle='--', linewidth=2, label='Obj Dual (Dual Infeasibility)')
plt.plot(iterations, obj_opt, marker='^', linestyle='-.', linewidth=2, label='Obj Opt (Optimality Gap)')

# 添加轴标签和标题
plt.xlabel('BCD Iterations', fontsize=12, fontweight='bold')
plt.ylabel('Objective Values / Infeasibility', fontsize=12, fontweight='bold')
plt.title('Convergence Analysis of Agent_NN_BCD', fontsize=14, pad=15)

# 设置刻度
plt.xticks(iterations)
plt.grid(True, which='both', linestyle=':', alpha=0.7)

# 添加图例
plt.legend(loc='best', frameon=True, shadow=True)

# 优化布局
plt.tight_layout()

# 保存并展示
plt.savefig('result/fig/BCD_convergence_plot.png', dpi=1000)
plt.show()