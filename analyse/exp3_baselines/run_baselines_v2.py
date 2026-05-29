import os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim, joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

def load_fixed_data():
    X_train = np.load('../../split_x_train.npy'); y_train = np.load('../../split_y_train.npy')
    X_val = np.load('../../split_x_val.npy'); y_val = np.load('../../split_y_val.npy')
    X_test = np.load('../../split_x_test.npy'); y_test = np.load('../../split_y_test.npy')
    sb = joblib.load('../../fitted_scaler_body.pkl'); sc = joblib.load('../../fitted_scaler_cloth.pkl')
    xtr = np.hstack([sb.transform(X_train[:,:6]), sc.transform(X_train[:,6:])])
    xva = np.hstack([sb.transform(X_val[:,:6]), sc.transform(X_val[:,6:])])
    xte = np.hstack([sb.transform(X_test[:,:6]), sc.transform(X_test[:,6:])])
    def ml(x,y,s): return DataLoader(TensorDataset(torch.tensor(x,dtype=torch.float32), torch.tensor(y,dtype=torch.float32).unsqueeze(1)), batch_size=64, shuffle=s)
    return (xtr,y_train,xva,y_val,xte,y_test), ml(xtr,y_train,True), ml(xva,y_val,False), ml(xte,y_test,False)

(xtr,ytr,xva,yva,xte,yte), tr_l, va_l, te_l = load_fixed_data()
res = []
d = xtr.shape[1]

xtrv = np.vstack([xtr,xva]); ytrv = np.concatenate([ytr,yva])
ps = PredefinedSplit(np.concatenate([-1*np.ones(len(xtr)), np.zeros(len(xva))]))

rf = GridSearchCV(RandomForestRegressor(random_state=2024), {'n_estimators':[100]}, cv=ps, scoring='r2', n_jobs=-1).fit(xtrv, ytrv)
rp = rf.predict(xte)
r2 = r2_score(yte,rp); mse = mean_squared_error(yte,rp); sp, _ = spearmanr(yte,rp)
res.append({'Model':'Random Forest','R2':r2,'MSE':mse,'Spearman':sp})
print(f'  RF  R2={r2:.4f}  MSE={mse:.4f}  Spearman={sp:.4f}')

svr = GridSearchCV(SVR(), {'C':[1,10]}, cv=ps, scoring='r2', n_jobs=-1).fit(xtrv, ytrv)
sp_pred = svr.predict(xte)
r2 = r2_score(yte,sp_pred); mse = mean_squared_error(yte,sp_pred); sp, _ = spearmanr(yte,sp_pred)
res.append({'Model':'SVR','R2':r2,'MSE':mse,'Spearman':sp})
print(f'  SVR  R2={r2:.4f}  MSE={mse:.4f}  Spearman={sp:.4f}')

def train_dl(model, epochs=60):
    criterion = nn.HuberLoss()
    opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=20, T_mult=2)
    br = -float('inf'); bs = None
    for _ in range(epochs):
        model.train()
        for x, y in tr_l:
            opt.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        model.eval(); vp, vy = [], []
        with torch.no_grad():
            for x, y in va_l: vp.extend(model(x.to(device)).cpu().numpy()); vy.extend(y.numpy())
        r = r2_score(vy, vp)
        if r > br: br = r; bs = model.state_dict()
    model.load_state_dict(bs); model.eval(); tp, ty = [], []
    with torch.no_grad():
        for x, y in te_l: tp.extend(model(x.to(device)).cpu().numpy()); ty.extend(y.numpy())
    r2 = r2_score(ty, tp); mse = mean_squared_error(ty, tp)
    sp, _ = spearmanr(np.array(ty).ravel(), np.array(tp).ravel())
    return r2, mse, sp

class MLP(nn.Module):
    def __init__(self): super().__init__(); self.n = nn.Sequential(nn.Linear(d,128), nn.ReLU(), nn.Linear(128,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x): return self.n(x)

r2,mse,sp = train_dl(MLP().to(device))
res.append({'Model':'MLP','R2':r2,'MSE':mse,'Spearman':sp})
print(f'  MLP  R2={r2:.4f}  MSE={mse:.4f}  Spearman={sp:.4f}')

class CNN1D(nn.Module):
    def __init__(self): super().__init__(); self.c = nn.Sequential(nn.Conv1d(1,16,3,1), nn.ReLU()); self.f = nn.Linear(16*(d-2), 1)
    def forward(self, x): c = self.c(x.unsqueeze(1)); return self.f(c.view(c.size(0), -1))

r2,mse,sp = train_dl(CNN1D().to(device), epochs=12)
res.append({'Model':'1D-CNN','R2':r2,'MSE':mse,'Spearman':sp})
print(f'  CNN  R2={r2:.4f}  MSE={mse:.4f}  Spearman={sp:.4f}')

class LSTMModel(nn.Module):
    def __init__(self): super().__init__(); self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2); self.f = nn.Linear(64, 1)
    def forward(self, x): o, _ = self.lstm(x.unsqueeze(-1)); return self.f(o[:, -1, :])

r2,mse,sp = train_dl(LSTMModel().to(device))
res.append({'Model':'LSTM','R2':r2,'MSE':mse,'Spearman':sp})
print(f'  LSTM  R2={r2:.4f}  MSE={mse:.4f}  Spearman={sp:.4f}')

res.append({'Model':'Ours(Benchmark)','R2':0.8258,'MSE':0.6173,'Spearman':0.8793})
df = pd.DataFrame(res); df.to_csv('baselines_metrics.csv', index=False)
print('\nFINAL:'); print(df.to_string(index=False))

idx = [i for i,m in enumerate(df.Model) if 'Ours' in m][0]
colors = ['#D3D3D3']*len(df); colors[idx] = '#FF6B6B'
plt.figure(figsize=(12,6))
plt.bar(df['Model'], df['R2'], color=colors)
for i,v in enumerate(df['R2']): plt.text(i, v+0.005, f'{v:.4f}', ha='center', fontweight='bold')
plt.title('Baseline Comparison'); plt.ylabel('R2'); plt.xticks(rotation=30)
plt.tight_layout(); plt.savefig('baselines_comparison.png', dpi=300); plt.close()
print('Done.')
