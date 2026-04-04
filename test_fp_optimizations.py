"""
简单单元测试：验证 identify_trusted_mask 优化后的行为
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from src.feasibility_pump import identify_trusted_mask

def test_identify_trusted_mask():
    """测试 identify_trusted_mask 的自适应阈值行为"""
    ng, T = 3, 24

    # 场景1：整体整数性强（mean_integrality < 0.15）
    x_LP_strong = np.zeros((ng, T))
    x_LP_strong[0, :12] = 0.02   # 接近0
    x_LP_strong[0, 12:] = 0.98   # 接近1
    x_LP_strong[1, :] = 0.01
    x_LP_strong[2, :] = 0.99

    x_init_k = np.round(x_LP_strong).astype(int)
    x_init_k_m = np.stack([x_init_k, x_init_k], axis=1)

    mask_strong = identify_trusted_mask(x_LP_strong, x_init_k, x_init_k_m)
    print(f"场景1（整数性强）- 受信任变量比例: {np.mean(mask_strong):.3f}")
    assert np.mean(mask_strong) > 0.90, "整数性强时应标记更多可信变量为真"

    # 场景2：整体整数性中等（mean_integrality ≈ 0.20）
    x_LP_medium = np.ones((ng, T)) * 0.5 + np.random.rand(ng, T) * 0.30 - 0.15
    x_LP_medium = np.clip(x_LP_medium, 0.0, 1.0)

    x_init_k_medium = np.round(x_LP_medium).astype(int)
    x_init_k_m_medium = np.stack([x_init_k_medium, x_init_k_medium], axis=1)

    mask_medium = identify_trusted_mask(x_LP_medium, x_init_k_medium, x_init_k_m_medium)
    print(f"场景2（整数性中等）- 受信任变量比例: {np.mean(mask_medium):.3f}")

    # 场景3：测试 adaptive_threshold=False
    mask_fixed = identify_trusted_mask(x_LP_strong, x_init_k, x_init_k_m, adaptive_threshold=False)
    print(f"场景3（固定阈值）- 受信任变量比例: {np.mean(mask_fixed):.3f}")

    # 场景4：提供 x_surr_lp
    x_surr_lp = x_LP_strong + np.random.rand(ng, T) * 0.05
    x_surr_lp = np.clip(x_surr_lp, 0.0, 1.0)

    mask_with_surr = identify_trusted_mask(x_LP_strong, x_init_k, x_init_k_m, x_surr_lp=x_surr_lp)
    print(f"场景4（含Surrogate LP）- 受信任变量比例: {np.mean(mask_with_surr):.3f}")

    print("\n所有测试通过！")
    return True

if __name__ == '__main__':
    try:
        test_identify_trusted_mask()
        print("\n[SUCCESS] identify_trusted_mask 优化验证通过")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
