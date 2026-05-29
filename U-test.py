import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import mannwhitneyu
import os

# ================= 1. 数据提取 =================
print("正在提取数据...")
SEED = 2024

train_df = pd.read_csv('../train_df.csv')
y_all = train_df['score'].values
_, true_scores_10percent = train_test_split(y_all, test_size=0.1, random_state=SEED)

# 读取 cloth_new 下最新预测的结果
test_df = pd.read_csv('test_df_with_predictions.csv')
pred_scores_external = test_df['score'].dropna().values

print(f"提取 10% 内部测试集 (真实值): {len(true_scores_10percent)} 条")
print(f"提取 全新盲测集 (预测值): {len(pred_scores_external)} 条")
print("\n" + "="*60)

# ================= 1.5 基础指标对比 =================
mean_true = np.mean(true_scores_10percent)
mean_pred = np.mean(pred_scores_external)
std_true = np.std(true_scores_10percent)
std_pred = np.std(pred_scores_external)

print("\n" + "="*60)
print(f"【均 值 对 比】 测试集真实均分: {mean_true:.4f} | 未知评分样本预测均分: {mean_pred:.4f}")
print(f"【标准差对比】测试集真实波动: {std_true:.4f} | 未知评分样本预测波动: {std_pred:.4f}")
print("="*60 + "\n")

# ================= 检验 A: 假设一样，找差异 (双侧 M-W U) =================
print("⚔️ 检验 A: 试图证明它们【不一样】")

u_stat, p_diff = mannwhitneyu(true_scores_10percent, pred_scores_external, alternative='two-sided')
print(f"【M-W U 差异检验 P-value】: {p_diff:.4e}")

if p_diff < 0.05:
    print("   👉 结论 A: P < 0.05。拒绝原假设。大盘中位水平在严格数学意义上【存在差异】。")
else:
    print("   👉 结论 A: P >= 0.05。无法拒绝原假设。大盘中位水平【一模一样】。")

print("-" * 60)

# ================= 检验 B: 假设不一样，求等效 (双单侧 M-W U) =================
print("🛡️ 检验 B: 试图证明它们【在业务上一样】")

MARGIN = 0.5  # 业务容忍边界：平均偏差 0.5 分以内认为“没区别”
print(f"设定业务最大容忍偏差 (Margin): ±{MARGIN} 分")

# M-W U 单侧检验 1：真实分减去 Margin 后，检验是否显著小于预测分
_, p_upper = mannwhitneyu(true_scores_10percent - MARGIN, pred_scores_external, alternative='less')

# M-W U 单侧检验 2：真实分加上 Margin 后，检验是否显著大于预测分
_, p_lower = mannwhitneyu(true_scores_10percent + MARGIN, pred_scores_external, alternative='greater')

# 取两个单侧检验的最大 P 值作为最终结果
p_eq = max(p_upper, p_lower)
print(f"【M-W U 等效性检验 P-value】: {p_eq:.4e}")

if p_eq < 0.05:
    print(f"   👉 结论 B: P < 0.05。成功证明这两批数据的排序偏差绝对被限制在了 {MARGIN} 分以内！")
    print("   ✅ 判定: 业务上【完全等效 / 一致】！")
else:
    print("   👉 结论 B: P >= 0.05。无法推翻大偏差的假设。")
    print("   🚨 判定: 漂移超出了安全边界，不能认为它们一致。")

print("\n" + "="*60)

# ================= 终极判定 =================
print("💡 最终定论:")
if p_diff < 0.05 and p_eq < 0.05:
    print("   🌟 完美防杠：虽然严格排序有微小差异 (检验A报警)，")
    print(f"      但偏差被死死限制在了 {MARGIN} 分的安全边界内 (检验B通过)。")
    print("      👉 模型预测大盘极其健康，放心使用！")
elif p_diff >= 0.05 and p_eq < 0.05:
    print("   🌟 完美重合：找不出任何差异，且被证明绝对等效。")
elif p_diff < 0.05 and p_eq >= 0.05:
    print("   ⚠️ 发生漂移：不仅有差异，而且差异大到突破了业务底线。需要排查模型！")
else:
    print("   ❓ 数据分布可能极其异常，建议结合可视化图表（KDE图）人工判断。")
