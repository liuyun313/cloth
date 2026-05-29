import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

df = pd.read_csv('internal_test_results.csv')
y_true = df['Actual'].values
y_pred = df['Predicted'].values

r2 = r2_score(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rho, p = spearmanr(y_true, y_pred)
print(f'R2={r2:.4f}  MSE={mse:.4f}  Spearman={rho:.4f}  p={p:.4e}')

plt.figure(figsize=(8,8))
plt.scatter(y_true, y_pred, alpha=0.6, color='dodgerblue', edgecolors='w', s=60)
vmin = min(y_true.min(), y_pred.min())
vmax = max(y_true.max(), y_pred.max())
plt.plot([vmin, vmax], [vmin, vmax], 'r--', lw=2, label='y=x')
plt.title(f'Internal Test: Predicted vs Actual\nR2={r2:.4f}  Spearman={rho:.4f}', fontsize=14)
plt.xlabel('Actual Score'); plt.ylabel('Predicted Score')
plt.legend(); plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout(); plt.savefig('prediction_distribution.png', dpi=300); plt.close()
print('prediction_distribution.png saved')
