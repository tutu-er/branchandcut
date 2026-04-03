特例归档：active_sets_case3lite_T24_n1000_20260403_180137.json
============================================================

生成命令（仓库根目录执行）：
  python run_active_set.py

对应脚本配置（run_active_set.py 顶部）：
  CASE_NAME = "case3lite"
  HORIZON = 24
  MAX_SAMPLES = 1000
  TARGET_SAMPLES = 1000
  ALPHA = 0.70
  DELTA = 0.05
  EPSILON = 0.10
  T_DELTA = 1.0
  PARALLEL = False
  OUTPUT_PATH = None

说明：OUTPUT_PATH 为 None 时，ActiveSetLearner.save_active_sets_json 会按
  active_sets_{case}_T{T}_n{n}_{timestamp}.json
写入 result/active_set/。本文件为 2026-04-03 18:01:37 左右生成的一次完整运行结果。

求解器：Gurobi（UC MIP），算例负荷由 case_registry.build_case3lite_base_load 构造。
