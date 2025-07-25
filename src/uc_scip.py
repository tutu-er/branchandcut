import numpy as np
import pandas as pd

import os
os.environ["PATH"] = r"d:\apps\scipoptsuite-9.2.3\bin;" + os.environ["PATH"]
from pyscipopt import Model, quicksum, multidict, SCIP_PARAMSETTING, SCIP_EVENTTYPE
from pypower.makePTDF import makePTDF
from pypower.ext2int import ext2int
from pypower.idx_gen import GEN_BUS, PMIN, PMAX
from pypower.idx_brch import RATE_A

from pathlib import Path
import io
import sys
import re

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.case39_pypower import get_case39_pypower

def print_basic_info(model):
    """打印基本求解信息"""
    print("\n=== 基本求解信息 ===")
    
    try:
        print(f"总节点数: {model.getNTotalNodes()}")
    except:
        print("总节点数: N/A")

def get_lp_solution_info(model):
    """获取LP松弛解信息"""
    print("\n=== LP松弛解信息 ===")
    
    try:
        if model.getStage() >= 4:  # SCIP_STAGE_SOLVING
            print(f"当前LP目标值: {model.getLPObjVal()}")
    except:
        print("当前LP目标值: N/A")
        
    try:
        print(f"LP行数: {model.getNLPRows()}")
    except:
        print("LP行数: N/A")
        
    try:
        print(f"LP列数: {model.getNLPCols()}")
    except:
        print("LP列数: N/A")
        
    try:
        print(f"LP迭代数: {model.getNLPIterations()}")
    except:
        print("LP迭代数: N/A")

def analyze_constraints(model):
    """分析约束信息"""
    print("\n=== 约束信息 ===")
    
    try:
        print(f"当前约束数: {model.getNConss()}")
    except:
        print("当前约束数: N/A")
    
    try:
        # 获取约束类型统计
        cons_types = {}
        for cons in model.getConss():
            # 尝试获取约束类型的安全方法
            cons_type = get_constraint_type_safe(cons)
            cons_types[cons_type] = cons_types.get(cons_type, 0) + 1
        
        print("约束类型分布:")
        for cons_type, count in cons_types.items():
            print(f"  {cons_type}: {count}")
    except Exception as e:
        print(f"约束分析失败: {e}")

def get_constraint_type_safe(constraint):
    """安全地获取约束类型"""
    try:
        # 尝试不同的方法获取约束类型
        if hasattr(constraint, 'getType'):
            return constraint.getType()
        elif hasattr(constraint, 'getConstype'):
            return constraint.getConstype()
        elif hasattr(constraint, 'name'):
            # 根据约束名称推断类型
            name = constraint.name if constraint.name else "unknown"
            if 'linear' in name.lower() or 'c' in name.lower():
                return 'linear'
            elif 'bound' in name.lower():
                return 'bounddisjunction'
            else:
                return 'unknown'
        else:
            return 'unknown'
    except Exception as e:
        return f'error_{str(e)[:20]}'

def get_constraint_attribute_safe(constraint, attr_name, default_value=None):
    """安全地获取约束属性"""
    try:
        if hasattr(constraint, attr_name):
            return getattr(constraint, attr_name)()
        else:
            return default_value
    except Exception as e:
        return f'error_{str(e)[:20]}'

def save_problem_to_file(model, filename):
    """保存问题到文件用于分析"""
    try:
        model.writeProblem(filename)
        print(f"问题已保存到: {filename}")
    except Exception as e:
        print(f"保存问题文件失败: {e}")

class UnitCommitmentModelSCIP:
    def __init__(self, ppc, Pd, T_delta):
        self.ppc = ppc
        ppc = ext2int(ppc)
        self.baseMVA = ppc['baseMVA']
        self.bus = ppc['bus']
        self.gen = ppc['gen']
        self.branch = ppc['branch']
        self.gencost = ppc['gencost']
        self.Pd = Pd
        self.T_delta = T_delta
        self.T = Pd.shape[1]
        self.ng = self.gen.shape[0]
        self.nb = self.branch.shape[0]
        self.model = Model("UnitCommitment")
        self.model.setParam('display/verblevel', 4)
        
        self._build_model()

    def _build_model(self):
        # 变量
        self.pg = {}
        self.x = {}
        self.coc = {}
        self.cpower = {}
        for g in range(self.ng):
            for t in range(self.T):
                self.pg[g, t] = self.model.addVar(lb=0, name=f"pg_{g}_{t}")
                self.x[g, t] = self.model.addVar(vtype="B", name=f"x_{g}_{t}")
                self.cpower[g, t] = self.model.addVar(lb=0, name=f"cpower_{g}_{t}")
            for t in range(self.T-1):
                self.coc[g, t] = self.model.addVar(lb=0, name=f"coc_{g}_{t}")

        # 有功平衡和出力上下界
        for t in range(self.T):
            self.model.addCons(
                quicksum(self.pg[g, t] for g in range(self.ng)) == np.sum(self.Pd[:, t]),
                name=f'power_balance_{t}'
            )
            for g in range(self.ng):
                self.model.addCons(self.pg[g, t] >= self.gen[g, PMIN] * self.x[g, t])
                self.model.addCons(self.pg[g, t] <= self.gen[g, PMAX] * self.x[g, t])

        # 爬坡约束
        Ru = 0.4 * self.gen[:, PMAX] / self.T_delta
        Rd = 0.4 * self.gen[:, PMAX] / self.T_delta
        Ru_co = 0.3 * self.gen[:, PMAX]
        Rd_co = 0.3 * self.gen[:, PMAX]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addCons(
                    self.pg[g, t] - self.pg[g, t-1] <= Ru[g] * self.x[g, t-1] + Ru_co[g] * (1 - self.x[g, t-1])
                )
                self.model.addCons(
                    self.pg[g, t-1] - self.pg[g, t] <= Rd[g] * self.x[g, t] + Rd_co[g] * (1 - self.x[g, t])
                )

        # 最小开机/关机时间约束
        Ton = int(4 * self.T_delta)
        Toff = int(4 * self.T_delta)
        for g in range(self.ng):
            for t in range(1, Ton+1):
                for t1 in range(self.T - t):
                    self.model.addCons(self.x[g, t1+1] - self.x[g, t1] <= self.x[g, t1+t])
            for t in range(1, Toff+1):
                for t1 in range(self.T - t):
                    self.model.addCons(-self.x[g, t1+1] + self.x[g, t1] <= 1 - self.x[g, t1+t])

        # 启停成本
        start_cost = self.gencost[:, 1]
        shut_cost = self.gencost[:, 2]
        for t in range(1, self.T):
            for g in range(self.ng):
                self.model.addCons(self.coc[g, t-1] >= -start_cost[g] * (self.x[g, t] - self.x[g, t-1]))
                self.model.addCons(self.coc[g, t-1] >= shut_cost[g] * (self.x[g, t-1] - self.x[g, t]))
                self.model.addCons(self.coc[g, t-1] >= 0)

        # 发电成本
        for t in range(self.T):
            for g in range(self.ng):
                self.model.addCons(
                    self.cpower[g, t] >= self.gencost[g, -2]/self.T_delta * self.pg[g, t] + self.gencost[g, -1]/self.T_delta * self.x[g, t]
                )

        # 潮流约束
        try:
            nb = self.Pd.shape[0]
            G = np.zeros((nb, self.ng))
            for g in range(self.ng):
                bus_idx = int(self.gen[g, GEN_BUS])
                G[bus_idx, g] = 1
            PTDF = makePTDF(self.baseMVA, self.bus, self.branch)
            branch_limit = self.branch[:, RATE_A]
            for t in range(self.T):
                for l in range(self.branch.shape[0]):
                    expr = quicksum(
                        (PTDF[l, :] @ G[:, g]) * self.pg[g, t] for g in range(self.ng)
                    )
                    const_term = PTDF[l, :] @ self.Pd[:, t]
                    self.model.addCons(expr - const_term <= branch_limit[l])
                    self.model.addCons(-expr + const_term <= branch_limit[l])
        except ImportError:
            print('未安装pypower，DCPF潮流约束未添加。')

        # 目标函数
        obj = quicksum(self.cpower[g, t] for g in range(self.ng) for t in range(self.T)) \
            + quicksum(self.coc[g, t] for g in range(self.ng) for t in range(self.T-1))
        self.model.setObjective(obj, "minimize")

    def configure_scip_parameters(self, strategy="default"):
        """配置SCIP参数
        
        Args:
            strategy: 求解策略
                - "default": 默认平衡策略
                - "fast": 快速求解策略
                - "quality": 高质量解策略
                - "aggressive": 激进策略（更多割平面和启发式）
        """
        print(f"=== 配置SCIP参数 (策略: {strategy}) ===")
        
        if strategy == "fast":
            # 快速求解策略
            self.model.setParam('presolving/maxrounds', 10)
            self.model.setParam('separating/maxrounds', 3)
            self.model.setParam('separating/maxroundsroot', 5)
            self.model.setParam('heuristics/rins/freq', -1)  # 禁用
            self.model.setParam('limits/time', 600)  # 10分钟
            print("配置为快速求解模式")
            
        elif strategy == "quality":
            # 高质量解策略
            self.model.setParam('presolving/maxrounds', 100)
            self.model.setParam('separating/maxrounds', 20)
            self.model.setParam('separating/maxroundsroot', 50)
            self.model.setParam('separating/maxcuts', 5000)
            self.model.setParam('heuristics/rins/freq', 10)
            self.model.setParam('limits/time', 7200)  # 2小时
            print("配置为高质量解模式")
            
        elif strategy == "aggressive":
            # 激进策略
            self.model.setParam('presolving/maxrounds', 200)
            self.model.setParam('separating/maxrounds', 30)
            self.model.setParam('separating/maxroundsroot', 100)
            self.model.setParam('separating/maxcuts', 10000)
            self.model.setParam('separating/maxcutsroot', 20000)
            self.model.setParam('heuristics/rins/freq', 5)
            self.model.setParam('limits/time', 10800)  # 3小时
            print("配置为激进求解模式")
            
        else:  # default
            # 默认平衡策略
            self.model.setParam('presolving/maxrounds', 50)
            self.model.setParam('separating/maxrounds', 10)
            self.model.setParam('separating/maxroundsroot', 20)
            self.model.setParam('separating/maxcuts', 1000)
            self.model.setParam('separating/maxcutsroot', 2000)
            self.model.setParam('heuristics/rins/freq', 20)
            self.model.setParam('limits/time', 3600)  # 1小时
            print("配置为默认平衡模式")
        
        # 通用设置
        self.model.setParam('display/verblevel', 3)
        self.model.setParam('display/freq', 100)
        
        # 分支策略
        self.model.setParam('branching/relpscost/priority', 100000)
        self.model.setParam('branching/inference/priority', 50000)
        
        # 分离器配置
        separators = {
            'gomory': {'priority': 1000, 'freq': 1},
            'cmir': {'priority': 900, 'freq': 1},
            'clique': {'priority': 800, 'freq': 1},
            'impliedbounds': {'priority': 700, 'freq': 1},
            'strongcg': {'priority': 600, 'freq': 1}
        }
        
        for sep_name, config in separators.items():
            try:
                for param, value in config.items():
                    self.model.setParam(f'separating/{sep_name}/{param}', value)
            except Exception as e:
                print(f"配置分离器 {sep_name} 失败: {e}")
        
        # 启发式设置
        heuristics = {
            'rounding': {'freq': 1},
            'shifting': {'freq': 10},
            'localbranching': {'freq': 50}
        }
        
        for heur_name, config in heuristics.items():
            try:
                for param, value in config.items():
                    self.model.setParam(f'heuristics/{heur_name}/{param}', value)
            except Exception as e:
                print(f"配置启发式 {heur_name} 失败: {e}")
        
        # 并行设置
        try:
            import multiprocessing
            ncpus = min(4, multiprocessing.cpu_count())
            self.model.setParam('parallel/maxnthreads', ncpus)
            print(f"启用并行计算，使用 {ncpus} 个线程")
        except:
            print("无法启用并行计算")

    def solve(self, strategy="default"):
        """求解模型
        
        Args:
            strategy: 求解策略 ("default", "fast", "quality", "aggressive")
        """
        
        # 保存求解前的问题状态
        print("=== 开始求解 ===")
        print(f"求解前约束数: {self.model.getNConss()}")
        print(f"求解前变量数: {self.model.getNVars()}")
        
        # 配置SCIP参数
        self.configure_scip_parameters(strategy)
        
        # 开始优化
        print("\n=== 开始优化 ===")
        self.model.optimize()
        status = self.model.getStatus()
        
        # 打印求解统计信息
        self.print_solution_statistics()
        
        # 打印基本信息
        print_basic_info(self.model)
        get_lp_solution_info(self.model)
        analyze_constraints(self.model)
        
        if status == "optimal":
            pg_sol = np.zeros((self.ng, self.T))
            x_sol = np.zeros((self.ng, self.T))
            for g in range(self.ng):
                for t in range(self.T):
                    pg_sol[g, t] = self.model.getVal(self.pg[g, t])
                    x_sol[g, t] = self.model.getVal(self.x[g, t])
            print(f"总运行成本: {self.model.getObjVal()}")
            return pg_sol, x_sol, self.model.getObjVal()
        else:
            print(f"未找到最优解，状态: {status}")
            return None, None, None
    
    def print_solution_statistics(self):
        """打印详细的求解统计信息"""
        print("\n=== 求解统计信息 ===")
        
        try:
            print(f"求解时间: {self.model.getSolvingTime():.2f} 秒")
        except:
            print("求解时间: N/A")
            
        try:
            print(f"预求解时间: {self.model.getPresolvingTime():.2f} 秒")
        except:
            print("预求解时间: N/A")
            
        try:
            print(f"总节点数: {self.model.getNTotalNodes()}")
        except:
            print("总节点数: N/A")
            
        try:
            print(f"最优性间隙: {self.model.getGap():.4f}")
        except:
            print("最优性间隙: N/A")
            
        try:
            print(f"双界: {self.model.getDualbound():.2f}")
        except:
            print("双界: N/A")
            
        try:
            print(f"原界: {self.model.getPrimalbound():.2f}")
        except:
            print("原界: N/A")
    

if __name__ == "__main__":
    load_df = pd.read_csv('src/load.csv', header=None)
    Pd = load_df.values  # shape: (nb, T)
    T = Pd.shape[1]
    T_delta = 4  # 如有需要可根据数据调整

    ppc = get_case39_pypower()

    # 创建模型对象
    uc = UnitCommitmentModelSCIP(ppc, Pd, T_delta)
    pg_sol, x_sol, total_cost = uc.solve()

    if pg_sol is not None:
        pass
        # print("机组出力方案：", pg_sol)
        # print("机组启停方案：", x_sol)
        # print("总成本：", total_cost)

        # ====== 结果绘图 ======
        # import matplotlib.pyplot as plt
        # import seaborn as sns
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