import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, joblib

import random
SEED = 2024
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

def load_fixed_data():
    X_train = np.load('../../split_x_train.npy'); y_train = np.load('../../split_y_train.npy')
    X_val   = np.load('../../split_x_val.npy');   y_val   = np.load('../../split_y_val.npy')
    X_test  = np.load('../../split_x_test.npy');  y_test  = np.load('../../split_y_test.npy')
    sb = joblib.load('../../fitted_scaler_body.pkl'); sc = joblib.load('../../fitted_scaler_cloth.pkl')
    xb_tr, xc_tr = sb.transform(X_train[:,:6]), sc.transform(X_train[:,6:])
    xb_va, xc_va = sb.transform(X_val[:,:6]),   sc.transform(X_val[:,6:])
    xb_te, xc_te = sb.transform(X_test[:,:6]),  sc.transform(X_test[:,6:])
    def ml(xb, xc, y, s): return DataLoader(TensorDataset(torch.tensor(xb,dtype=torch.float32), torch.tensor(xc,dtype=torch.float32), torch.tensor(y,dtype=torch.float32).unsqueeze(1)), batch_size=64, shuffle=s)
    return ml(xb_tr,xc_tr,y_train,True), ml(xb_va,xc_va,y_val,False), ml(xb_te,xc_te,y_test,False)

tr_l, va_l, te_l = load_fixed_data()
nc = next(iter(tr_l))[1].shape[1]
print(f'Body:6 Cloth:{nc}')

class E(nn.Module):
    def __init__(self, n, d): super().__init__(); self.w = nn.Parameter(torch.randn(n, d)); self.b = nn.Parameter(torch.randn(n, d))
    def forward(self, x): return x.unsqueeze(-1) * self.w + self.b

class Net(nn.Module):
    def __init__(self, d, h, hd):
        super().__init__(); self.eb = E(6,d); self.ec = E(nc,d)
        self.ab = nn.MultiheadAttention(d, h, batch_first=True)
        self.ac = nn.MultiheadAttention(d, h, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d*2, hd), nn.BatchNorm1d(hd), nn.GELU(), nn.Dropout(0.2), nn.Linear(hd, 16), nn.GELU(), nn.Linear(16, 1))
    def forward(self, xb, xc):
        eb, ec = self.eb(xb), self.ec(xc)
        ab_o, _ = self.ab(eb, ec, ec); ac_o, _ = self.ac(ec, eb, eb)
        return self.mlp(torch.cat([self.n1(eb+ab_o).mean(1), self.n2(ec+ac_o).mean(1)], 1))

def train_eval(d, h, hd):
    model = Net(d, h, hd).to(device); opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2); criterion = nn.HuberLoss()
    br = -float('inf'); bs = None
    for _ in range(150):
        model.train()
        for bx, cx, y in tr_l: opt.zero_grad(); loss = criterion(model(bx.to(device), cx.to(device)), y.to(device)); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0); opt.step()
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
    sp, pv = spearmanr(np.array(ty).ravel(), np.array(tp).ravel())
    return r2, mse, sp

BENCH = {'Embed_Dim':16, 'Num_Heads':1, 'Hidden_Dim':64, 'R2':0.8258, 'MSE':0.6173, 'Spearman':0.8793}
res = []
cnt = 0
for d in [8, 16, 32, 64]:
    for h in [1, 2, 4]:
        for hd in [32, 64, 128]:
            cnt += 1; r2, mse, sp = train_eval(d, h, hd)
            res.append({'Embed_Dim':d, 'Num_Heads':h, 'Hidden_Dim':hd, 'R2':r2, 'MSE':mse, 'Spearman':sp})
            print(f'  [{cnt}/36] dim={d} heads={h} hidden={hd}  R2={r2:.4f}  MSE={mse:.4f}  Spearman={sp:.4f}')

res.append(BENCH)
df = pd.DataFrame(res)
df.to_csv('sensitivity_grid_results.csv', index=False)

grid = df.iloc[:-1]
best = grid.iloc[grid['R2'].idxmax()]
b_row = grid[(grid['Embed_Dim']==16) & (grid['Num_Heads']==1) & (grid['Hidden_Dim']==64)]
print(f'\nBEST grid: dim={int(best.Embed_Dim)} heads={int(best.Num_Heads)} hidden={int(best.Hidden_Dim)} R2={best.R2:.4f} Spearman={best.Spearman:.4f}')
print(f'Grid(16,1,64): R2={b_row.R2.values[0]:.4f} Spearman={b_row.Spearman.values[0]:.4f}')
print(f'BENCHMARK: R2=0.8258 Spearman=0.8793')

for param, name in [('Embed_Dim','Embedding Dimension'),('Num_Heads','Number of Heads'),('Hidden_Dim','Hidden Dimension')]:
    fig, ax1 = plt.subplots(figsize=(7,5)); ax2 = ax1.twinx()
    pivot = grid.groupby(param)['R2'].mean(); pivot_mse = grid.groupby(param)['MSE'].mean()
    ax1.plot(pivot.index.astype(str), pivot.values, 'b-o', lw=2, ms=8, label='R2')
    ax2.plot(pivot_mse.index.astype(str), pivot_mse.values, 'r-s', lw=2, ms=8, label='MSE')
    ax1.set_xlabel(param); ax1.set_ylabel('R2', color='b'); ax2.set_ylabel('MSE', color='r')
    plt.title(f'Parameter Sensitivity: {name}')
    plt.tight_layout(); plt.savefig(f'sensitivity_{param.lower()}.png', dpi=300); plt.close()
print('Done.')
