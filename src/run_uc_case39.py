import numpy as np
from case39_uc_data import get_case39_uc
from case39_pypower import get_case39_pypower
from uc_gurobipy import UnitCommitmentModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 假设 Pd 为 bus 数据中的 Pd 列，T_delta 需根据实际时间步长设置
# baseMVA, bus, gen, branch, gencost = get_case39_uc()


# 读取负荷数据 load.csv，假设每行为一个节点，每列为一个时段
load_df = pd.read_csv('src/load.csv', header=None)
Pd = load_df.values  # shape: (nb, T)
T = Pd.shape[1]
T_delta = 4  # 如有需要可根据数据调整

ppc = get_case39_pypower()

# 创建模型对象
uc = UnitCommitmentModel(ppc, Pd, T_delta)
# 求解
pg_sol, x_sol, total_cost = uc.solve()

if pg_sol is not None:
    uc.model.write('src/presolved.lp')
    print('已导出presolve后的模型文件：presolved.lp')
    pass
    # print("机组出力方案：", pg_sol)
    # print("机组启停方案：", x_sol)
    # print("总成本：", total_cost)

    # ====== 结果绘图 ======
    # # 机组出力折线图
    # plt.figure(figsize=(12, 6))
    # for g in range(pg_sol.shape[0]):
    #     if np.sum(x_sol[g, :]) > 0:
    #         plt.plot(range(1, pg_sol.shape[1]+1), pg_sol[g, :], label=f'机组{g+1}')
    # plt.xlabel('时段')
    # plt.ylabel('出力 (MW)')
    # plt.title('机组出力折线图')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # # 启停状态热力图
    # plt.figure(figsize=(12, 4))
    # sns.heatmap(x_sol, cmap='Blues', cbar=False)
    # plt.xlabel('时段')
    # plt.ylabel('机组编号')
    # plt.title('机组启停状态热力图 (蓝色=运行, 白色=停机)')
    # plt.tight_layout()
    # plt.show()
else:
    print("未找到可行解")