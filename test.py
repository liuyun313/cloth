import pandas as pd, numpy as np, torch, torch.nn as nn, joblib, os
from torch.utils.data import DataLoader, TensorDataset

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
            nn.Linear(64, 16), nn.GELU(), nn.Linear(16, 1))
    def forward(self, x_body, x_cloth):
        emb_body = self.embedder_body(x_body)
        emb_style = self.embedder_cloth(x_cloth)
        attn_body_style, _ = self.cross_attn_body_to_style(emb_body, emb_style, emb_style)
        attn_style_body, _ = self.cross_attn_style_to_body(emb_style, emb_body, emb_body)
        out_body = self.norm1(emb_body + attn_body_style)
        out_style = self.norm2(emb_style + attn_style_body)
        merged = torch.cat([out_body.mean(dim=1), out_style.mean(dim=1)], dim=1)
        return self.mlp_head(merged)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[Test] Device: {device}')

test_df = pd.read_csv('../test_df.csv')
body_cols = ['VHI','Across Shoulder','Bust','Waist','Low Hip','Inseam']
cloth_cols = ['Cloth Category','Shoulder','Torso - Bust','Sil - Bust','Torso - Waist','Sil - Waist','Torso - Hip','Sil - Hip','Hem','Top Hue','Top Saturation','Top Value','Bottom Hue','Bottom Saturation','Bottom Value']

X_test_body_raw = test_df[body_cols].values
X_test_cloth_df = test_df[cloth_cols]
numeric_cloth_cols = X_test_cloth_df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cloth_cols = X_test_cloth_df.select_dtypes(include=['object','category']).columns.tolist()
X_test_cloth_categorical = pd.get_dummies(X_test_cloth_df[categorical_cloth_cols], dummy_na=False, dtype=int)
loaded_dummy_cols = joblib.load('train_dummy_columns.pkl')
X_test_cloth_categorical = X_test_cloth_categorical.reindex(columns=loaded_dummy_cols, fill_value=0)
X_test_cloth_raw = pd.concat([X_test_cloth_df[numeric_cloth_cols], X_test_cloth_categorical], axis=1).values

scaler_body = joblib.load('fitted_scaler_body.pkl')
scaler_cloth = joblib.load('fitted_scaler_cloth.pkl')
X_test_body_scaled = scaler_body.transform(X_test_body_raw)
X_test_cloth_scaled = scaler_cloth.transform(X_test_cloth_raw)

test_loader = DataLoader(TensorDataset(
    torch.tensor(X_test_body_scaled, dtype=torch.float32),
    torch.tensor(X_test_cloth_scaled, dtype=torch.float32)), batch_size=64, shuffle=False)

nb, nc = X_test_body_scaled.shape[1], X_test_cloth_scaled.shape[1]
model = TabularCrossAttentionNet(nb, nc, embed_dim=16, num_heads=1).to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

preds = []
with torch.no_grad():
    for bx, cx in test_loader:
        preds.extend(model(bx.to(device), cx.to(device)).cpu().numpy().flatten())

test_df['score'] = preds
test_df.to_csv('test_df_with_predictions.csv', index=False)
print(f'Done. {len(preds)} samples saved to test_df_with_predictions.csv')
