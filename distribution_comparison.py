import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

internal = pd.read_csv('internal_test_results.csv')
external = pd.read_csv('test_df_with_predictions.csv')
y_int_true = internal['Actual'].values
y_ext_pred = external['score'].values

print(f'Internal true: n={len(y_int_true)} mean={y_int_true.mean():.2f} std={y_int_true.std():.2f}')
print(f'External pred: n={len(y_ext_pred)} mean={y_ext_pred.mean():.2f} std={y_ext_pred.std():.2f}')

xmin = min(y_int_true.min(), y_ext_pred.min())
xmax = max(y_int_true.max(), y_ext_pred.max())
x_grid = np.linspace(xmin, xmax, 300)

kde_int = gaussian_kde(y_int_true)
kde_ext = gaussian_kde(y_ext_pred)

plt.figure(figsize=(12, 7))
plt.plot(x_grid, kde_int(x_grid), color='steelblue', lw=2.5, label=f'Internal Test True Labels (n={len(y_int_true)})')
plt.fill_between(x_grid, kde_int(x_grid), alpha=0.15, color='steelblue')
plt.plot(x_grid, kde_ext(x_grid), color='coral', lw=2.5, label=f'External Test Predicted Labels (n={len(y_ext_pred)})')
plt.fill_between(x_grid, kde_ext(x_grid), alpha=0.15, color='coral')

plt.axvline(y_int_true.mean(), color='steelblue', linestyle='--', lw=1.5, alpha=0.7)
plt.axvline(y_ext_pred.mean(), color='coral', linestyle='--', lw=1.5, alpha=0.7)

plt.title('Label Distribution Comparison: Internal True vs External Predicted', fontsize=14, fontweight='bold')
plt.xlabel('Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=300)
plt.close()
print('distribution_comparison.png saved')
