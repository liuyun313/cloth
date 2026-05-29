import pandas as pd
import numpy as np

# 1. 读取原始数据
df = pd.read_csv('train_df.csv')

# 2. 映射所需要的特定列名
feature_cols = [
    'VHI', 'Across Shoulder', 'Bust', 'Waist', 'Low Hip', 'Inseam', 
    'Cloth Category', 'Shoulder', 'Torso - Bust', 'Sil - Bust', 
    'Torso - Waist', 'Sil - Waist', 'Torso - Hip', 'Sil - Hip', 
    'Hem', 'Top Hue', 'Top Saturation', 'Top Value', 
    'Bottom Hue', 'Bottom Saturation', 'Bottom Value'
]
target_col = 'score'

# 提取所需的特征子集和标签
X_subset = df[feature_cols]
y = df[target_col]

# 3. 区分数值型与类别型特征
numeric_cols = X_subset.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_subset.select_dtypes(include=['object', 'category']).columns.tolist()

# 4. 执行 One-Hot 编码 (这里只有 Cloth Category 需要)
df_categorical_onehot = pd.get_dummies(X_subset[categorical_cols], dummy_na=False, dtype=int)

# 5. 特征拼接 (横向拼接)
final_features = pd.concat([X_subset[numeric_cols], df_categorical_onehot], axis=1)

# --- 打印结果核对 ---
print(f"提取完成！")
print(f"用到的数值特征 (保留原始值): {len(numeric_cols)} 个")
print(f"用到的类别特征 (One-Hot处理): {len(categorical_cols)} 个")
print(f"拼接后的特征矩阵 X 维度: {final_features.shape} (20个数值特征 + 展开后的Cloth Category)")
print(f"标签向量 y 维度: {y.shape}")

# 6. 保存为 Numpy 数组格式 (.npy)
np.save('features_X_subset.npy', final_features.values)
np.save('labels_y_subset.npy', y.values)

print("提取的数据已成功保存为 features_X_subset.npy 和 labels_y_subset.npy！")