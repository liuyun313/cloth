import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import random
import os
import joblib

# ================= 0. 全局配置 =================
SEED = 2024
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [Train] 运行设备: {device}")

# ================= 1. 数据预处理与 8:1:1 划分 =================
print("读取训练数据 train_df.csv ...")
train_df = pd.read_csv('../train_df.csv')

body_cols = ['VHI', 'Across Shoulder', 'Bust', 'Waist', 'Low Hip', 'Inseam']
cloth_cols = ['Cloth Category', 'Shoulder', 'Torso - Bust', 'Sil - Bust', 
              'Torso - Waist', 'Sil - Waist', 'Torso - Hip', 'Sil - Hip', 
              'Hem', 'Top Hue', 'Top Saturation', 'Top Value', 
              'Bottom Hue', 'Bottom Saturation', 'Bottom Value']
target_col = 'score'

y = train_df[target_col].values

# 处理 Body 特征 (全为连续值)
X_body_raw = train_df[body_cols].values
num_features_body = X_body_raw.shape[1]

# 处理 Cloth 特征
X_cloth_df = train_df[cloth_cols]
numeric_cloth_cols = X_cloth_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cloth_cols = X_cloth_df.select_dtypes(include=['object', 'category']).columns.tolist()

X_cloth_categorical = pd.get_dummies(X_cloth_df[categorical_cloth_cols], dummy_na=False, dtype=int)
train_dummy_columns = X_cloth_categorical.columns.tolist()
joblib.dump(train_dummy_columns, 'train_dummy_columns.pkl')

X_cloth_raw = pd.concat([X_cloth_df[numeric_cloth_cols], X_cloth_categorical], axis=1).values
num_features_cloth = X_cloth_raw.shape[1]

# 组合方便一起划分
X_combined = np.hstack((X_body_raw, X_cloth_raw))

# 严谨的 8:1:1 划分 (80%训练, 10%验证, 10%内部盲测)
X_temp, X_test, y_temp, y_test = train_test_split(X_combined, y, test_size=0.1, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/9, random_state=SEED)

# ================= 保存划分后的原始数据集 =================
np.save('split_x_train.npy', X_train)
np.save('split_y_train.npy', y_train)
np.save('split_x_val.npy', X_val)
np.save('split_y_val.npy', y_val)
np.save('split_x_test.npy', X_test)
np.save('split_y_test.npy', y_test)
print("💾 划分好的底层数据集 (Train/Val/Test) 已固化保存至 .npy 文件，确保后续所有实验数据绝对一致！")

# 分离 Body 和 Cloth
X_train_body = X_train[:, :num_features_body]
X_train_cloth = X_train[:, num_features_body:]
X_val_body = X_val[:, :num_features_body]
X_val_cloth = X_val[:, num_features_body:]
X_test_body = X_test[:, :num_features_body]
X_test_cloth = X_test[:, num_features_body:]

# 标准化 Body
scaler_body = StandardScaler()
X_train_body_scaled = scaler_body.fit_transform(X_train_body)
X_val_body_scaled = scaler_body.transform(X_val_body)
X_test_body_scaled = scaler_body.transform(X_test_body)
joblib.dump(scaler_body, 'fitted_scaler_body.pkl')

# 标准化 Cloth
scaler_cloth = StandardScaler()
X_train_cloth_scaled = scaler_cloth.fit_transform(X_train_cloth)
X_val_cloth_scaled = scaler_cloth.transform(X_val_cloth)
X_test_cloth_scaled = scaler_cloth.transform(X_test_cloth)
joblib.dump(scaler_cloth, 'fitted_scaler_cloth.pkl')

def to_loader(X_body_np, X_cloth_np, y_np, batch_size=64, shuffle=False):
    return DataLoader(
        TensorDataset(
            torch.tensor(X_body_np, dtype=torch.float32),
            torch.tensor(X_cloth_np, dtype=torch.float32), 
            torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)), 
        batch_size=batch_size, shuffle=shuffle)

train_loader = to_loader(X_train_body_scaled, X_train_cloth_scaled, y_train, shuffle=True)
val_loader = to_loader(X_val_body_scaled, X_val_cloth_scaled, y_val, shuffle=False)
test_loader = to_loader(X_test_body_scaled, X_test_cloth_scaled, y_test, shuffle=False)

# ================= 2. 模型定义 =================
class NumericalEmbedder(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, embed_dim))
        self.bias = nn.Parameter(torch.randn(num_features, embed_dim))
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight + self.bias

class TabularCrossAttentionNet(nn.Module):
    def __init__(self, num_body_features, num_cloth_features, embed_dim=16, num_heads=1):
        super().__init__()
        self.embedder_body = NumericalEmbedder(num_body_features, embed_dim)
        self.embedder_cloth = NumericalEmbedder(num_cloth_features, embed_dim)
        
        self.cross_attn_body_to_style = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.cross_attn_style_to_body = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 1)
        )
        
    def forward(self, x_body, x_cloth):
        emb_body = self.embedder_body(x_body)
        emb_style = self.embedder_cloth(x_cloth)
        
        attn_body_style, _ = self.cross_attn_body_to_style(emb_body, emb_style, emb_style)
        attn_style_body, _ = self.cross_attn_style_to_body(emb_style, emb_body, emb_body)
        
        out_body = self.norm1(emb_body + attn_body_style)
        out_style = self.norm2(emb_style + attn_style_body)
        
        merged = torch.cat([out_body.mean(dim=1), out_style.mean(dim=1)], dim=1)
        return self.mlp_head(merged)

# ================= 3. 模型训练 =================
model = TabularCrossAttentionNet(num_features_body, num_features_cloth, embed_dim=16, num_heads=1).to(device)
criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

epochs = 150
best_val_r2 = -float('inf')

print("\n--- 开始训练 ---")
for epoch in range(epochs):
    model.train()
    for batch_body, batch_cloth, batch_y in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(batch_body.to(device), batch_cloth.to(device)), batch_y.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    scheduler.step()
    
    # 验证集评估
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch_body, batch_cloth, batch_y in val_loader:
            val_preds.extend(model(batch_body.to(device), batch_cloth.to(device)).cpu().numpy())
            val_targets.extend(batch_y.numpy())
            
    val_r2 = r2_score(val_targets, val_preds)
    
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        torch.save(model.state_dict(), 'best_model.pth')
        
    if (epoch + 1) % 15 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Val R2: {val_r2:.4f} | 最佳 Val R2: {best_val_r2:.4f}")

# ================= 4. 内部盲测与绘图 =================
print("\n" + "="*50)
print("加载最佳模型权重，进行内部 Test 集 (10%数据) 盲测与绘图...")

model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

test_preds, test_targets = [], []
with torch.no_grad():
    for batch_body, batch_cloth, batch_y in test_loader:
        test_preds.extend(model(batch_body.to(device), batch_cloth.to(device)).cpu().numpy())
        test_targets.extend(batch_y.numpy())

final_test_r2 = r2_score(test_targets, test_preds)
final_test_mse = mean_squared_error(test_targets, test_preds)
final_test_spearman, final_test_spearman_p = spearmanr(
    np.array(test_targets).ravel(), np.array(test_preds).ravel()
)

print(f"✅ 内部盲测 Test MSE: {final_test_mse:.4f}")
print(f"✅ 内部盲测 Test R2 Score: {final_test_r2:.4f}")
print(f"✅ 内部盲测 Test Spearman rho: {final_test_spearman:.4f} (p={final_test_spearman_p:.4e})")
print("="*50)

# ----- 保存内部测试集结果供绘图使用 -----
results_df = pd.DataFrame({
    'Actual': np.array(test_targets).flatten(),
    'Predicted': np.array(test_preds).flatten()
})
results_df.to_csv('internal_test_results.csv', index=False)
print("✅ 内部盲测结果已保存为 'internal_test_results.csv'，你可以使用 plot_scatter.py 单独绘图。")
print("✅ 训练脚本全部执行完毕！")
