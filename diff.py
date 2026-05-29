import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp, entropy
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. 数据提取 =================
print("正在提取数据...")
SEED = 2024

# 1. 提取内部 10% 测试集的【真实分数】(注意路径调整为上一级)
train_df = pd.read_csv('../train_df.csv')
y_all = train_df['score'].values

_, true_scores_10percent = train_test_split(y_all, test_size=0.1, random_state=SEED)

# 2. 提取全新盲测集的【预测分数】(读取当前目录下新模型的预测结果)
test_df = pd.read_csv('test_df_with_predictions.csv')
pred_scores_external = test_df['score'].dropna().values

print(f"提取 10% 内部测试集 (真实值): {len(true_scores_10percent)} 条")
print(f"提取 全新盲测集 (预测值): {len(pred_scores_external)} 条")


# ================= 2. 核心统计学检验 (K-S & KL 两个指标) =================
print("\n" + "="*50)
print("📊 分布差异检验报告: K-S 检验 与 KL 散度")
print("="*50)

# --- 指标 1: K-S 检验 (双样本检验，验证是否属于同一分布) ---
ks_stat, p_value = ks_2samp(true_scores_10percent, pred_scores_external)
print(f"【K-S 检验统计量】: {ks_stat:.4f}")
print(f"【K-S 检验 P-value】: {p_value:.4e}")

if p_value < 0.05:
    print("   ⚠️ K-S 结论: P-value < 0.05。拒绝原假设，这两批分数的分布【存在显著差异】。")
else:
    print("   ✅ K-S 结论: P-value >= 0.05。两者【属于同一个分布】，大盘形态一致。")

print("-" * 50)

# --- 指标 2: KL 散度 (以 1 分为区间严格分箱，衡量信息差) ---
min_score = np.floor(min(min(true_scores_10percent), min(pred_scores_external)))
max_score = np.ceil(max(max(true_scores_10percent), max(pred_scores_external)))
bins = np.arange(min_score, max_score + 2, 1)  # 强制 1 分为 1 个区间

hist_true, _ = np.histogram(true_scores_10percent, bins=bins)
hist_pred, _ = np.histogram(pred_scores_external, bins=bins)

# 平滑处理防止 log(0) 报错
epsilon = 1e-10
p_true = (hist_true + epsilon) / np.sum(hist_true + epsilon)
q_pred = (hist_pred + epsilon) / np.sum(hist_pred + epsilon)

kl_divergence = entropy(p_true, q_pred)
print(f"【KL 散度 (区间=1)】: {kl_divergence:.4f}")

if kl_divergence < 0.1:
    print("   🌟 KL 结论: < 0.1，预测分数在每个细分区间上的占比，与真实分布【高度吻合】。")
else:
    print("   🚨 KL 结论: >= 0.1，预测分数与历史真实形态存在【一定的偏差】。")
print("="*50)


# ================= 3. 可视化对比图 =================
print("\n正在生成分布对比图...")
plt.figure(figsize=(10, 6))

# 保持原始颜色
sns.kdeplot(true_scores_10percent, fill=True, color="blue", label="Internal 10% Test Set (True Scores)", alpha=0.5)
sns.kdeplot(pred_scores_external, fill=True, color="orange", label="External Blind Test (Predicted Scores)", alpha=0.5)

plt.title('Distribution: 10% Internal True Scores vs External Predicted Scores', fontsize=14, pad=15)
plt.xlabel('Fashion Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('distribution_ks_kl_original_colors.png', dpi=300)
print("🎯 图片已保存为 'distribution_ks_kl_original_colors.png'！")


# ================= 4. 导出原始分数 (Raw Scores) =================
print("\n正在导出原始分数数据表...")

# 使用 pd.Series 自动对齐长度不一致的数据。缺少的行会自动填充为空白 (NaN)
raw_df = pd.DataFrame({
    'Internal_True_Score': pd.Series(true_scores_10percent),
    'External_Predicted_Score': pd.Series(pred_scores_external)
})

csv_filename = 'raw_scores_export.csv'
raw_df.to_csv(csv_filename, index=False)

print(f"📊 原始分数表已成功导出为: '{csv_filename}'")
print("表格头部预览：")
print(raw_df.head())
