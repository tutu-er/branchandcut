import numpy as np
import gurobipy as gp
from gurobipy import GRB
import sys
import io
import re
# 使用 case118 数据（pypower 内置）
from pypower.api import case118
# 修正：引入正确的成本索引
from pypower.idx_cost import MODEL, NCOST, COST
from pypower.idx_bus import PD, BUS_TYPE, REF
from pypower.idx_gen import PMAX, PMIN, GEN_BUS
from pypower.idx_brch import F_BUS, T_BUS, BR_X, RATE_A

# ==========================================
# 0. 解析 Gurobi 输出日志中的割平面统计
# ==========================================
def parse_gurobi_cut_statistics(log_output):
    """
    从 Gurobi 的输出日志中解析割平面统计信息
    
    Args:
        log_output: Gurobi 的输出日志字符串
        
    Returns:
        dict: 割平面类型和数量的字典，例如 {'Gomory': 19, 'MIR': 2, ...}
    """
    cut_stats = {}
    
    # 方法：逐行解析，更可靠
    lines = log_output.split('\n')
    in_cutting_section = False
    
    for line in lines:
        # 检测 "Cutting planes:" 标题
        if 'Cutting planes:' in line:
            in_cutting_section = True
            continue
        
        if in_cutting_section:
            # 检查是否是割平面统计行
            # 格式: "  Type: count" 或 "  Type-name: count" 或 "  Type name: count"
            match = re.match(r'\s+([\w\s-]+):\s+(\d+)', line)
            if match:
                cut_type = match.group(1).strip()
                count = int(match.group(2))
                cut_stats[cut_type] = count
            elif line.strip() == '':
                # 空行，继续（可能还有更多割平面信息）
                continue
            elif not line.startswith(' ') and line.strip():
                # 非缩进行且非空，说明割平面部分结束
                break
    
    return cut_stats

# ==========================================
# 1. 定义 Callback 用于记录割平面 (Cuts)
# ==========================================
class CutTracker:
    def __init__(self):
        self.cut_counts = 0  # 回调中观察到的最大割平面数
        self.max_cutcnt = 0   # 历史最大值
        self.callback_called = False
        self.callback_where = []
        self.cutcnt_history = []  # 记录所有割平面数量变化
    
    def callback(self, model, where):
        self.callback_called = True
        self.callback_where.append(where)
        
        # 尝试在 MIP 回调中获取割平面数量
        if where == GRB.Callback.MIP:
            try:
                # MIP_CUTCNT: 当前节点中活跃的割平面数量
                cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
                self.cutcnt_history.append(('MIP', cutcnt))
                if cutcnt > self.max_cutcnt:
                    self.max_cutcnt = cutcnt
                    self.cut_counts = cutcnt
            except Exception as e:
                pass
        
        # 也在 MIPNODE 回调中尝试
        elif where == GRB.Callback.MIPNODE:
            try:
                cutcnt = model.cbGet(GRB.Callback.MIP_CUTCNT)
                self.cutcnt_history.append(('MIPNODE', cutcnt))
                if cutcnt > self.max_cutcnt:
                    self.max_cutcnt = cutcnt
                    self.cut_counts = cutcnt
            except Exception as e:
                pass

# ==========================================
# 2. UC 问题构建 (Case118, 更紧迫的机组需求)
# ==========================================
def solve_uc_case118():
    print("\n" + "="*40)
    print("构建 Case118 24时段 UC 问题 (机组需求紧迫)...")
    print("="*40)

    # --- 数据准备 ---
    mpc = case118()
    baseMVA = mpc['baseMVA']
    
    # 1. 发电机数据
    gens = mpc['gen']
    n_gens = gens.shape[0]
    gen_buses = gens[:, GEN_BUS].astype(int) - 1
    p_max = gens[:, PMAX]
    p_min = gens[:, PMIN]
    
    # 2. 修正：从 gencost 提取成本，并增加成本差异
    # gencost 格式: [MODEL, STARTUP, SHUTDOWN, NCOST, COST...]
    gencost = mpc['gencost']
    
    c1 = np.zeros(n_gens)
    c0 = np.zeros(n_gens)
    start_cost = np.zeros(n_gens)  # 启动成本
    shut_cost = np.zeros(n_gens)   # 停机成本
    
    for i in range(n_gens):
        model_type = gencost[i, MODEL]
        n_cost = int(gencost[i, NCOST])
        
        # 提取启停成本
        start_cost[i] = gencost[i, 1] if gencost.shape[1] > 1 else 100.0
        shut_cost[i] = gencost[i, 2] if gencost.shape[1] > 2 else 50.0
        
        # 定位成本系数开始的列索引
        start_idx = COST 
        
        if model_type == 2: # 多项式模型
            if n_cost == 3: # 二次函数: c2, c1, c0
                c1[i] = gencost[i, start_idx + 1]
                c0[i] = gencost[i, start_idx + 2]
            elif n_cost == 2: # 线性函数: c1, c0
                c1[i] = gencost[i, start_idx]
                c0[i] = gencost[i, start_idx + 1]
        else:
            # 默认为0或简单处理
            c1[i] = 10.0
            c0[i] = 0.0
    
    # 增加成本差异：使不同机组的成本差异更大
    # 根据机组容量大小设置不同的成本结构
    # 大机组：边际成本低，固定成本高（投资大但运行效率高）
    # 小机组：边际成本高，固定成本低（投资小但运行效率低）
    
    # 按容量排序索引
    capacity_order = np.argsort(p_max)
    
    # 边际成本（c1）：大机组低，小机组高（范围：0.3x 到 2.5x）
    c1_multiplier = np.linspace(0.3, 2.5, n_gens)
    c1_multiplier = c1_multiplier[capacity_order]  # 按容量顺序分配
    np.random.shuffle(c1_multiplier)  # 随机打乱，增加多样性
    c1 = c1 * c1_multiplier
    
    # 固定成本（c0）：大机组高，小机组低（范围：0.2x 到 3.0x）
    # 固定成本差异应该更明显，因为这是影响机组选择的关键因素
    c0_multiplier = np.linspace(0.2, 3.0, n_gens)
    c0_multiplier = c0_multiplier[::-1]  # 反转，使大机组固定成本高
    np.random.shuffle(c0_multiplier)  # 随机打乱
    c0 = c0 * c0_multiplier
    
    # 如果原始固定成本太小或为0，设置一个基于容量的基础值
    # 固定成本应该与机组容量相关
    base_fixed_cost = p_max * 0.1  # 基础固定成本：容量的10%
    c0 = np.maximum(c0, base_fixed_cost * c0_multiplier)
    
    # 启停成本：与固定成本相关，大机组启停成本高
    start_cost_multiplier = c0_multiplier * 0.5  # 启动成本约为固定成本的0.5倍
    shut_cost_multiplier = c0_multiplier * 0.3   # 停机成本约为固定成本的0.3倍
    start_cost = start_cost * start_cost_multiplier
    shut_cost = shut_cost * shut_cost_multiplier
    
    # 确保成本为正且合理
    c1 = np.maximum(c1, 0.01)
    c0 = np.maximum(c0, 1.0)  # 固定成本至少为1.0
    start_cost = np.maximum(start_cost, 10.0)
    shut_cost = np.maximum(shut_cost, 5.0)
    
    # # 打印成本统计信息（用于调试）
    # print(f"\n成本差异统计:")
    # print(f"  边际成本 (c1): 最小={c1.min():.4f}, 最大={c1.max():.4f}, 比例={c1.max()/c1.min():.2f}x")
    # print(f"  固定成本 (c0): 最小={c0.min():.4f}, 最大={c0.max():.4f}, 比例={c0.max()/c0.min():.2f}x")
    # print(f"  启动成本: 最小={start_cost.min():.4f}, 最大={start_cost.max():.4f}, 比例={start_cost.max()/start_cost.min():.2f}x")

    # 3. 线路数据
    branches = mpc['branch']
    n_lines = branches.shape[0]
    f_bus = branches[:, F_BUS].astype(int) - 1
    t_bus = branches[:, T_BUS].astype(int) - 1
    x_reactance = branches[:, BR_X]
    rate_a = branches[:, RATE_A] * 0.05
    # rate_a[rate_a == 0] = 9999  # 处理无限容量

    # 4. 负荷数据
    buses = mpc['bus']
    n_buses = buses.shape[0]
    load_p0 = buses[:, PD]
    ref_bus = np.where(buses[:, BUS_TYPE] == REF)[0][0]

    # 24小时负荷系数 - 使用更紧迫的负荷曲线（高峰更高，低谷不低）
    T = 24
    load_profile = np.array([
        0.75, 0.70, 0.65, 0.65, 0.70, 0.80, 0.90, 0.95, 
        1.0, 1.05, 1.15, 1.20, 1.15, 1.20, 1.15, 1.10,
        1.05, 1.10, 1.25, 1.20, 1.10, 1.0, 0.90, 0.80
    ]) * 1.2
    
    # 使问题更紧迫：减少发电机容量（乘以0.85，使容量更紧张）
    p_max = p_max
    
    # --- 模型建立 (MILP) ---
    m = gp.Model("UC_Case118")
    m.setParam('OutputFlag', 1)
    # 确保割平面生成是启用的
    m.setParam('Cuts', 2)  # 2 = 激进割平面生成
    # # 设置求解时间限制（case118 规模更大，可能需要更长时间）
    m.setParam('TimeLimit', 600)  # 10分钟
    
    # 变量
    p_gen = m.addVars(n_gens, T, lb=0, name="P")
    u_gen = m.addVars(n_gens, T, lb=0, ub=1, vtype=GRB.BINARY, name="u") # 机组状态
    coc = m.addVars(n_gens, T-1, lb=0, name="coc")  # 启停成本变量
    theta = m.addVars(n_buses, T, lb=-2*np.pi, ub=2*np.pi, name="theta")
    
    # T_delta: 每个时段的时间长度（小时）
    T_delta = 1.0  # 假设每个时段1小时

    # 约束
    for t in range(T):
        # 1. 出力上下限
        for g in range(n_gens):
            m.addConstr(p_gen[g, t] <= p_max[g] * u_gen[g, t])
            m.addConstr(p_gen[g, t] >= p_min[g] * u_gen[g, t])

        # 2. 参考节点
        m.addConstr(theta[ref_bus, t] == 0)

        # 3. 线路潮流 (DC)
        for l in range(n_lines):
            i, j = f_bus[l], t_bus[l]
            flow = (theta[i, t] - theta[j, t]) / x_reactance[l] * baseMVA
            m.addConstr(flow <= rate_a[l])
            m.addConstr(flow >= -rate_a[l])

        # 4. 节点功率平衡
        for b in range(n_buses):
            # 机组注入
            gen_idx = np.where(gen_buses == b)[0]
            p_in = gp.quicksum(p_gen[g, t] for g in gen_idx)
            # 负荷
            p_out_load = load_p0[b] * load_profile[t]
            # 线路流出
            p_out_flow = 0
            # 作为起始节点流出
            idx_from = np.where(f_bus == b)[0]
            for l in idx_from:
                p_out_flow += (theta[b, t] - theta[t_bus[l], t]) / x_reactance[l] * baseMVA
            # 作为终止节点流入 (即负的流出)
            idx_to = np.where(t_bus == b)[0]
            for l in idx_to:
                p_out_flow -= (theta[f_bus[l], t] - theta[b, t]) / x_reactance[l] * baseMVA
            
            m.addConstr(p_in - p_out_load == p_out_flow)
    
    # 5. 爬坡约束
    Ru = 0.4 * p_max / T_delta  # 上爬坡速率 (MW/h)
    Rd = 0.4 * p_max / T_delta  # 下爬坡速率 (MW/h)
    Ru_co = 0.3 * p_max  # 冷启动上爬坡速率
    Rd_co = 0.3 * p_max  # 冷启动下爬坡速率
    
    for t in range(1, T):
        for g in range(n_gens):
            # 上爬坡约束：如果上一时段开机，用正常爬坡速率；如果关机，用冷启动速率
            m.addConstr(p_gen[g, t] - p_gen[g, t-1] <= Ru[g] * u_gen[g, t-1] + Ru_co[g] * (1 - u_gen[g, t-1]))
            # 下爬坡约束：如果当前时段开机，用正常爬坡速率；如果关机，用冷启动速率
            m.addConstr(p_gen[g, t-1] - p_gen[g, t] <= Rd[g] * u_gen[g, t] + Rd_co[g] * (1 - u_gen[g, t]))
    
    # 6. 最小开机时间和最小关机时间约束
    Ton = int(4 * T_delta)  # 最小开机时间：4小时
    Toff = int(4 * T_delta)  # 最小关机时间：4小时
    
    for g in range(n_gens):
        # 最小开机时间约束：如果机组在t1+1时刻启动，则必须在t1到t1+t期间保持运行
        for t in range(1, Ton+1):
            for t1 in range(T - t):
                m.addConstr(u_gen[g, t1+1] - u_gen[g, t1] <= u_gen[g, t1+t])
        
        # 最小关机时间约束：如果机组在t1+1时刻停机，则必须在t1到t1+t期间保持停机
        for t in range(1, Toff+1):
            for t1 in range(T - t):
                m.addConstr(-u_gen[g, t1+1] + u_gen[g, t1] <= 1 - u_gen[g, t1+t])
    
    # 7. 启停成本约束
    for t in range(1, T):
        for g in range(n_gens):
            # 启动成本：如果从t-1到t启动，则产生启动成本
            m.addConstr(coc[g, t-1] >= start_cost[g] * (u_gen[g, t] - u_gen[g, t-1]))
            # 停机成本：如果从t-1到t停机，则产生停机成本
            m.addConstr(coc[g, t-1] >= shut_cost[g] * (u_gen[g, t-1] - u_gen[g, t]))
            # 非负约束
            m.addConstr(coc[g, t-1] >= 0)
    
    # 目标：最小化运行成本 + 空载成本 + 启停成本
    obj = 0
    for t in range(T):
        for g in range(n_gens):
            obj += c1[g] * p_gen[g, t] + c0[g] * u_gen[g, t]
    # 添加启停成本
    for t in range(T-1):
        for g in range(n_gens):
            obj += coc[g, t]
    m.setObjective(obj, GRB.MINIMIZE)

    # --- 求解 ---
    tracker = CutTracker()
    
    # 捕获 Gurobi 的输出日志
    old_stdout = sys.stdout
    log_capture = io.StringIO()
    sys.stdout = log_capture
    
    m.optimize(tracker.callback)
    
    # 恢复标准输出
    sys.stdout = old_stdout
    log_output = log_capture.getvalue()
    
    # 解析日志中的割平面信息
    cut_stats = parse_gurobi_cut_statistics(log_output)
    total_cuts_from_log = sum(cut_stats.values())
    
    print(f"UC 求解状态: {m.Status}")
    if m.SolCount > 0:
        print(f"UC 最优成本: {m.ObjVal:.2f}")
    
    # 打印割平面统计信息
    if cut_stats:
        print(f"\n从 Gurobi 日志解析的割平面统计:")
        for cut_type, count in cut_stats.items():
            print(f"  {cut_type}: {count}")
        print(f"  总计: {total_cuts_from_log}")
    else:
        print("\n未能从日志中解析割平面信息")
    
    # 使用日志中的总割平面数作为统计
    return n_gens * T, total_cuts_from_log

# ==========================================
# 3. 0-1 背包问题 (同规模对比，多约束版本以触发割平面)
# ==========================================
def solve_knapsack_similar(target_vars):
    print("\n" + "="*40)
    print(f"构建同规模 0-1 背包问题 (N={target_vars})...")
    print("="*40)
    
    np.random.seed(42)
    weights = np.random.randint(10, 100, target_vars)
    values = np.random.randint(50, 500, target_vars)
    
    # 使用多约束背包问题（更复杂，更容易触发割平面）
    # 创建多个资源约束，模拟多维度背包问题
    n_constraints = max(3, target_vars // 400)  # 根据规模调整约束数
    capacities = []
    constraint_weights = []
    
    for c in range(n_constraints):
        # 每个约束有不同的权重分布
        np.random.seed(42 + c)
        c_weights = np.random.randint(5, 80, target_vars)
        constraint_weights.append(c_weights)
        # 容量设置为总重量的30-40%
        capacities.append(int(sum(c_weights) * (0.3 + 0.1 * c / n_constraints)))

    m = gp.Model("Knapsack")
    m.setParam('OutputFlag', 1)
    # 确保割平面生成是启用的
    m.setParam('Cuts', 2)  # 2 = 激进割平面生成
    # 减少预求解，避免问题被过度简化
    m.setParam('Presolve', 1)  # 1 = 保守预求解（保留更多结构）
    # 增加割平面轮数
    m.setParam('CutPasses', 5)  # 根节点割平面轮数
    m.setParam('CutAggPasses', 3)  # 激进割平面轮数

    x = m.addVars(target_vars, vtype=GRB.BINARY, name="x")

    # 目标函数
    m.setObjective(gp.quicksum(values[i] * x[i] for i in range(target_vars)), GRB.MAXIMIZE)
    
    # 添加多个资源约束（多维度背包）
    for c in range(n_constraints):
        m.addConstr(gp.quicksum(constraint_weights[c][i] * x[i] for i in range(target_vars)) <= capacities[c],
                   name=f"capacity_constraint_{c}")
    
    # 添加额外的逻辑约束，增加问题复杂度
    # 例如：某些物品不能同时选择
    n_conflicts = min(20, target_vars // 10)
    for conf in range(n_conflicts):
        i1 = np.random.randint(0, target_vars)
        i2 = np.random.randint(0, target_vars)
        if i1 != i2:
            m.addConstr(x[i1] + x[i2] <= 1, name=f"conflict_{conf}")

    tracker = CutTracker()
    
    # 捕获 Gurobi 的输出日志
    old_stdout = sys.stdout
    log_capture = io.StringIO()
    sys.stdout = log_capture
    
    m.optimize(tracker.callback)
    
    # 恢复标准输出
    sys.stdout = old_stdout
    log_output = log_capture.getvalue()
    
    # 解析日志中的割平面信息
    cut_stats = parse_gurobi_cut_statistics(log_output)
    total_cuts_from_log = sum(cut_stats.values())

    print(f"背包 求解状态: {m.Status}")
    if m.SolCount > 0:
        print(f"背包 最优值: {m.ObjVal:.2f}")
    
    # 打印割平面统计信息
    if cut_stats:
        print(f"\n从 Gurobi 日志解析的割平面统计:")
        for cut_type, count in cut_stats.items():
            print(f"  {cut_type}: {count}")
        print(f"  总计: {total_cuts_from_log}")
    else:
        print("\n未能从日志中解析割平面信息")
        print("提示: 如果问题太简单，Gurobi 可能不需要割平面即可求解")
    
    # 使用日志中的总割平面数作为统计
    return total_cuts_from_log

# ==========================================
# 4. 主执行逻辑
# ==========================================
if __name__ == "__main__":
    try:
        # 求解 UC
        uc_vars, uc_cuts = solve_uc_case118()
        
        # 求解 背包（规模与 UC 问题的二进制变量数相同）
        kp_cuts = solve_knapsack_similar(uc_vars)
        
        print("\n" + "#"*50)
        print("  Gurobi 求解统计对比")
        print("#"*50)
        print(f"{'Problem':<15} | {'Binary Vars':<12} | {'Total Cuts':<10}")
        print("-" * 45)
        print(f"{'UC (Case118)':<15} | {uc_vars:<12} | {uc_cuts:<10}")
        print(f"{'Knapsack':<15} | {uc_vars:<12} | {kp_cuts:<10}")
        
    except Exception as e:
        print(f"Error: {e}")