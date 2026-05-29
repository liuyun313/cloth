import os
import random
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


SEED = 2024
DATA_PATH = "../train_df.csv"
OUTPUT_DIR = "attention_outputs"
TARGET_COL = "score"

BODY_COLS = ['VHI', 'Across Shoulder', 'Bust', 'Waist', 'Low Hip', 'Inseam']
CLOTH_COLS = ['Cloth Category', 'Shoulder', 'Torso - Bust', 'Sil - Bust', 
              'Torso - Waist', 'Sil - Waist', 'Torso - Hip', 'Sil - Hip', 
              'Hem', 'Top Hue', 'Top Saturation', 'Top Value', 
              'Bottom Hue', 'Bottom Saturation', 'Bottom Value']


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class NumericalEmbedder(nn.Module):
    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, embed_dim))
        self.bias = nn.Parameter(torch.randn(num_features, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight + self.bias


class TabularCrossAttentionNet(nn.Module):
    def __init__(self, num_body_features: int, num_cloth_features: int, embed_dim: int = 16, num_heads: int = 1):
        super().__init__()
        self.embedder_body = NumericalEmbedder(num_body_features, embed_dim)
        self.embedder_cloth = NumericalEmbedder(num_cloth_features, embed_dim)
        
        self.cross_attn_body_to_style = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.cross_attn_style_to_body = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 1)
        )

    def forward(self, x_body: torch.Tensor, x_cloth: torch.Tensor) -> torch.Tensor:
        pred, _, _ = self.forward_with_attention(x_body, x_cloth)
        return pred

    def forward_with_attention(
        self, x_body: torch.Tensor, x_cloth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb_body = self.embedder_body(x_body)
        emb_style = self.embedder_cloth(x_cloth)

        attn_body_style, w_body_style = self.cross_attn_body_to_style(
            emb_body, emb_style, emb_style, need_weights=True, average_attn_weights=False
        )
        attn_style_body, w_style_body = self.cross_attn_style_to_body(
            emb_style, emb_body, emb_body, need_weights=True, average_attn_weights=False
        )

        out_body = self.norm1(emb_body + attn_body_style)
        out_style = self.norm2(emb_style + attn_style_body)
        merged = torch.cat([out_body.mean(dim=1), out_style.mean(dim=1)], dim=1)
        pred = self.mlp_head(merged)
        return pred, w_body_style, w_style_body


def preprocess_with_alignment() -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET_COL].values

    # Body 特征
    x_body_raw = df[BODY_COLS].values
    body_names = BODY_COLS.copy()

    # Cloth 特征
    x_cloth_df = df[CLOTH_COLS]
    numeric_cloth_cols = x_cloth_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cloth_cols = x_cloth_df.select_dtypes(include=['object', 'category']).columns.tolist()

    x_cloth_categorical = pd.get_dummies(x_cloth_df[categorical_cloth_cols], dummy_na=False, dtype=int)
    dummy_cols = joblib.load("train_dummy_columns.pkl")
    x_cloth_categorical = x_cloth_categorical.reindex(columns=dummy_cols, fill_value=0)
    
    cloth_names = numeric_cloth_cols + list(x_cloth_categorical.columns)
    x_cloth_raw = pd.concat([x_cloth_df[numeric_cloth_cols], x_cloth_categorical], axis=1).values

    x_combined = np.hstack((x_body_raw, x_cloth_raw))
    num_body = x_body_raw.shape[1]

    # 切分取 10% 测试集用于分析
    _, x_test, _, y_test = train_test_split(x_combined, y, test_size=0.1, random_state=SEED)

    x_test_body = x_test[:, :num_body]
    x_test_cloth = x_test[:, num_body:]

    scaler_body = joblib.load("fitted_scaler_body.pkl")
    scaler_cloth = joblib.load("fitted_scaler_cloth.pkl")

    return scaler_body.transform(x_test_body), scaler_cloth.transform(x_test_cloth), y_test, body_names, cloth_names


def save_global_analysis(all_w_body_style: np.ndarray, all_w_style_body: np.ndarray, body_names: List[str], style_names: List[str]) -> None:
    # all_w_body_style shape: [N, heads, body_len, style_len]
    global_bs = all_w_body_style.mean(axis=(0, 1))
    global_sb = all_w_style_body.mean(axis=(0, 1))

    # 画图：Body -> Cloth (人体特征关注哪些服装特征)
    plt.figure(figsize=(max(10, len(style_names) * 0.4), max(6, len(body_names) * 0.5)))
    sns.heatmap(global_bs, xticklabels=style_names, yticklabels=body_names, cmap="Reds", cbar=True)
    plt.title("Global Attention: Body to Cloth")
    plt.xlabel("Cloth Features")
    plt.ylabel("Body Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "global_body_to_cloth_heatmap.png"), dpi=300)
    plt.close()

    # 画图：Cloth -> Body (服装特征关注哪些人体特征)
    plt.figure(figsize=(max(8, len(body_names) * 0.5), max(8, len(style_names) * 0.3)))
    sns.heatmap(global_sb, xticklabels=body_names, yticklabels=style_names, cmap="Blues", cbar=True)
    plt.title("Global Attention: Cloth to Body")
    plt.xlabel("Body Features")
    plt.ylabel("Cloth Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "global_cloth_to_body_heatmap.png"), dpi=300)
    plt.close()

    # 导出关联度最高的 Top CSV
    rows_bs = [(b, s, float(global_bs[i, j])) for i, b in enumerate(body_names) for j, s in enumerate(style_names)]
    pd.DataFrame(sorted(rows_bs, key=lambda x: x[2], reverse=True)[:50], columns=["Body_Feature", "Cloth_Feature", "Attention_Weight"]).to_csv(
        os.path.join(OUTPUT_DIR, "global_top_body_to_cloth.csv"), index=False, encoding="utf-8-sig")

    rows_sb = [(s, b, float(global_sb[i, j])) for i, s in enumerate(style_names) for j, b in enumerate(body_names)]
    pd.DataFrame(sorted(rows_sb, key=lambda x: x[2], reverse=True)[:50], columns=["Cloth_Feature", "Body_Feature", "Attention_Weight"]).to_csv(
        os.path.join(OUTPUT_DIR, "global_top_cloth_to_body.csv"), index=False, encoding="utf-8-sig")


def main() -> None:
    seed_everything(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Attention Analysis] 运行设备: {device}")

    x_test_body, x_test_cloth, y_test, body_names, cloth_names = preprocess_with_alignment()
    print(f"📦 [Data] 内部测试集样本数 = {len(y_test)}")
    print(f"🧬 [Feature] Body 特征数 = {len(body_names)} | Cloth 特征数 = {len(cloth_names)}")

    model = TabularCrossAttentionNet(num_body_features=x_test_body.shape[1], num_cloth_features=x_test_cloth.shape[1]).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    model.eval()

    tensor_body = torch.tensor(x_test_body, dtype=torch.float32, device=device)
    tensor_cloth = torch.tensor(x_test_cloth, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        preds_list, w_bs_list, w_sb_list = [], [], []
        batch_size = 256
        for i in range(0, len(y_test), batch_size):
            b_body, b_cloth = tensor_body[i:i+batch_size], tensor_cloth[i:i+batch_size]
            preds, w_bs, w_sb = model.forward_with_attention(b_body, b_cloth)
            preds_list.append(preds.cpu().numpy())
            w_bs_list.append(w_bs.cpu().numpy())
            w_sb_list.append(w_sb.cpu().numpy())
            
    w_bs_np = np.concatenate(w_bs_list, axis=0)
    w_sb_np = np.concatenate(w_sb_list, axis=0)

    print(f"\n📊 正在生成全局平均注意力热力图...")
    save_global_analysis(w_bs_np, w_sb_np, body_names, cloth_names)
    print(f"✅ 生成完毕！所有图片和表格均保存在 '{OUTPUT_DIR}/' 目录下。")

if __name__ == "__main__":
    main()
