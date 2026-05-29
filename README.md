# 服装合身度预测 - 双向交叉注意力网络

本项目通过建模人体尺寸（6个特征）与服装属性（17个特征）之间的交互关系，使用双向交叉注意力神经网络预测服装合身度评分。

## 第一步：配置环境

```
conda create -n cloth python=3.10 -y
conda activate cloth
pip install -r requirements.txt
```

## 第二步：特征提取

```
python feature_extract.py
```

处理原始数据，生成 train_df.csv（含特征和标签）以及 test_df.csv（含特征，无标签）。

## 第三步：模型训练

```
python train.py
```

- 将 train_df.csv 按 80%/10%/10% 划分为训练集/验证集/测试集
- 保存最优模型 best_model.pth

输出文件：best_model.pth, internal_test_results.csv


## 第四步：外部推理

```
python test.py
```

加载 best_model.pth，对 test_df.csv 进行预测，结果保存到 test_df_with_predictions.csv。

## 第五步：可视化与分析

```
python plot_scatter.py               # 预测 vs 真实散点图
python distribution_comparison.py    # KDE 密度分布对比图
python attention_analysis.py         # 注意力权重热力图
python diff.py                       # 子组评分布差异分析
python U-test.py                     # 统计显著性检验
```

## 文件说明

| 文件 | 作用 |
|------|------|
| feature_extract.py | 从原始数据提取特征，生成 train_df.csv 和 test_df.csv |
| train.py | 训练模型，保存最优权重和测试结果 |
| test.py | 加载训练好的模型，对外部测试集进行推理 |
| attention_analysis.py | 可视化注意力权重：Body<->Cloth 热力图 |
| plot_scatter.py | 内部测试集预测值 vs 真实值散点分布图 |
| distribution_comparison.py | 内部真实标签 vs 外部预测标签的 KDE 密度对比图 |
| diff.py | 不同子组间的评分布差异分析 |
| U-test.py | 类别间评分差异的 Mann-Whitney U 统计检验 |

