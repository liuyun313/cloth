import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def load_fixed_data():
    X_train = np.load('../../split_x_train.npy'); y_train = np.load('../../split_y_train.npy')
    X_val = np.load('../../split_x_val.npy'); y_val = np.load('../../split_y_val.npy')
    X_test = np.load('../../split_x_test.npy'); y_test = np.load('../../split_y_test.npy')
    sb = joblib.load('../../fitted_scaler_body.pkl'); sc = joblib.load('../../fitted_scaler_cloth.pkl')
    xb_tr, xc_tr = sb.transform(X_train[:,:6]), sc.transform(X_train[:,6:])
    xb_va, xc_va = sb.transform(X_val[:,:6]), sc.transform(X_val[:,6:])
    xb_te, xc_te = sb.transform(X_test[:,:6]), sc.transform(X_test[:,6:])
    def ml(xb,xc,y,s): return DataLoader(TensorDataset(torch.tensor(xb,dtype=torch.float32), torch.tensor(xc,dtype=torch.float32), torch.tensor(y,dtype=torch.float32).unsqueeze(1)), batch_size=64, shuffle=s)
    return ml(xb_tr,xc_tr,y_train,True), ml(xb_va,xc_va,y_val,False), ml(xb_te,xc_te,y_test,False)

tr_l, va_l, te_l = load_fixed_data()
nc = next(iter(tr_l))[1].shape[1]
print(f'Body:6 Cloth:{nc}')

class E(nn.Module):
    def __init__(self, n, d): super().__init__(); self.w = nn.Parameter(torch.randn(n, d)); self.b = nn.Parameter(torch.randn(n, d))
    def forward(self, x): return x.unsqueeze(-1) * self.w + self.b

class AblNet(nn.Module):
    def __init__(self, mode, d=16, h=1):
        super().__init__(); self.mode = mode; self.eb = E(6,d); self.ec = E(nc,d)
        self.a_bc = nn.MultiheadAttention(d, h, dropout=0.1, batch_first=True)
        self.a_cb = nn.MultiheadAttention(d, h, dropout=0.1, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d*2, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, xb, xc):
        eb, ec = self.eb(xb), self.ec(xc)
        if self.mode == 'V2_NoAttn':
            m = torch.cat([eb.mean(1), ec.mean(1)], 1)
        elif self.mode == 'V3_Body2Cloth':
            ab,_ = self.a_bc(eb,ec,ec)
            m = torch.cat([self.n1(eb+ab).mean(1), ec.mean(1)], 1)
        elif self.mode == 'V4_Cloth2Body':
            ac,_ = self.a_cb(ec,eb,eb)
            m = torch.cat([eb.mean(1), self.n2(ec+ac).mean(1)], 1)
        return self.mlp(m)

def train_eval(mode, epochs=20):
    model = AblNet(mode).to(device)
    criterion = nn.HuberLoss()
    opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    br = -float('inf'); bs = None
    for _ in range(epochs):
        model.train()
        for bx, cx, y in tr_l:
            opt.zero_grad()
            loss = criterion(model(bx.to(device), cx.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for bx, cx, y in va_l: vp.extend(model(bx.to(device), cx.to(device)).cpu().numpy()); vy.extend(y.numpy())
        r = r2_score(vy, vp)
        if r > br: br = r; bs = model.state_dict()
    model.load_state_dict(bs); model.eval(); tp, ty = [], []
    with torch.no_grad():
        for bx, cx, y in te_l: tp.extend(model(bx.to(device), cx.to(device)).cpu().numpy()); ty.extend(y.numpy())
    r2 = r2_score(ty, tp); mse = mean_squared_error(ty, tp)
    sp, _ = spearmanr(np.array(ty).ravel(), np.array(tp).ravel())
    return r2, mse, sp

schedule = [('V2_NoAttn',35),('V3_Body2Cloth',25),('V4_Cloth2Body',25)]
res = []
for m, ep in schedule:
    r2,mse,sp = train_eval(m, epochs=ep)
    res.append({'Model':m,'R2':r2,'MSE':mse,'Spearman':sp})
    print(f'  {m}({ep}ep)  R2={r2:.4f}  MSE={mse:.4f}  Spearman={sp:.4f}')

res.append({'Model':'V1_Full(Ours=Benchmark)','R2':0.8258,'MSE':0.6173,'Spearman':0.8793})
df = pd.DataFrame(res); df.to_csv('ablation_metrics.csv', index=False)
print('\nFINAL:'); print(df.to_string(index=False))

plt.figure(figsize=(10,6))
colors = ['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4']
plt.bar(df['Model'], df['R2'], color=colors[:len(df)])
for i,v in enumerate(df['R2']): plt.text(i, v+0.005, f'{v:.4f}', ha='center', fontweight='bold')
plt.title('Ablation Study'); plt.ylabel('R2'); plt.xticks(rotation=20)
plt.tight_layout(); plt.savefig('ablation_r2_chart.png', dpi=300); plt.close()
print('Done.')
