# 实验一：参数敏感性实验 (Parameter Sensitivity)

## 实验目的
探究模型两个核心深度学习超参数的变化对预测性能（R2 Score, MSE）的影响：
1. **Embedding Dimension (特征映射维度)**: 测试 [8, 16, 32, 64] 
   - 观察特征表示空间大小的极限，判断模型是否欠拟合或过拟合。
2. **Number of Heads (多头注意力头数)**: 测试 [1, 2, 4, 8]
   - 观察多头注意力机制对捕捉不同子空间特征组合的增益。

## 运行方式
确保在 HPA conda 环境下运行：
\`\`\`bash
cd exp1_sensitivity
python run_sensitivity.py
\`\`\`

## 输出产物
- \`sensitivity_results.csv\`: 记录所有参数组合的测试集指标。
- \`sensitivity_embed_dim.png\`: 维度敏感性折线图。
- \`sensitivity_heads.png\`: 注意力头数敏感性折线图。
