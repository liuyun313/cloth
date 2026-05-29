import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

SEED = 2024
DATA_PATH = "../train_df.csv"
BODY_COLS = ["VHI", "Across Shoulder", "Bust", "Waist", "Low Hip", "Inseam"]
CLOTH_COLS = ["Cloth Category", "Shoulder", "Torso - Bust", "Sil - Bust", 
              "Torso - Waist", "Sil - Waist", "Torso - Hip", "Sil - Hip", 
              "Hem", "Top Hue", "Top Saturation", "Top Value", 
              "Bottom Hue", "Bottom Saturation", "Bottom Value"]

def get_shared_data(batch_size=64):
    """
    统一的数据获取接口，保证所有实验(敏感性、消融、对比)的
    训练集(Train)、验证集(Val)和测试集(Test)完全一致。
    返回的 X 数据已经被相同的 Scaler 预处理过。
    """
    df = pd.read_csv(DATA_PATH)
    y = df["score"].values

    x_body_raw = df[BODY_COLS].values
    num_body = x_body_raw.shape[1]

    x_cloth_df = df[CLOTH_COLS]
    numeric_cloth_cols = x_cloth_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cloth_cols = x_cloth_df.select_dtypes(include=["object", "category"]).columns.tolist()

    x_cloth_categorical = pd.get_dummies(x_cloth_df[categorical_cloth_cols], dummy_na=False, dtype=int)
    # 强制对齐特征列，确保不管怎么跑，列都是一样的
    dummy_cols = joblib.load("../train_dummy_columns.pkl")
    x_cloth_categorical = x_cloth_categorical.reindex(columns=dummy_cols, fill_value=0)
    
    x_cloth_raw = pd.concat([x_cloth_df[numeric_cloth_cols], x_cloth_categorical], axis=1).values

    x_combined = np.hstack((x_body_raw, x_cloth_raw))

    # ================= 核心：严格锁定随机种子的切分 =================
    # 1. 拆分出 10% 作为 Test (完全等同于 train.py 的切分)
    x_temp, x_test, y_temp, y_test = train_test_split(x_combined, y, test_size=0.1, random_state=SEED)
    
    # 2. 剩下的拆分出 80% Train 和 10% Val (也就是 8:1 的比例)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=1/9, random_state=SEED)

    # 拆解 body 和 cloth
    x_train_body, x_train_cloth = x_train[:, :num_body], x_train[:, num_body:]
    x_val_body, x_val_cloth = x_val[:, :num_body], x_val[:, num_body:]
    x_test_body, x_test_cloth = x_test[:, :num_body], x_test[:, num_body:]

    # 直接复用 train.py 训练好的 Scaler 进行标准化
    scaler_body = joblib.load("../fitted_scaler_body.pkl")
    scaler_cloth = joblib.load("../fitted_scaler_cloth.pkl")

    x_train_body_scaled = scaler_body.transform(x_train_body)
    x_train_cloth_scaled = scaler_cloth.transform(x_train_cloth)
    x_val_body_scaled = scaler_body.transform(x_val_body)
    x_val_cloth_scaled = scaler_cloth.transform(x_val_cloth)
    x_test_body_scaled = scaler_body.transform(x_test_body)
    x_test_cloth_scaled = scaler_cloth.transform(x_test_cloth)

    # 给传统机器学习用的平铺特征
    ml_train_x = np.hstack((x_train_body_scaled, x_train_cloth_scaled))
    ml_val_x = np.hstack((x_val_body_scaled, x_val_cloth_scaled))
    ml_test_x = np.hstack((x_test_body_scaled, x_test_cloth_scaled))

    def make_loader(xb, xc, y_arr, shuffle):
        return DataLoader(TensorDataset(
            torch.tensor(xb, dtype=torch.float32),
            torch.tensor(xc, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)
        ), batch_size=batch_size, shuffle=shuffle)

    loaders = {
        "train": make_loader(x_train_body_scaled, x_train_cloth_scaled, y_train, True),
        "val": make_loader(x_val_body_scaled, x_val_cloth_scaled, y_val, False),
        "test": make_loader(x_test_body_scaled, x_test_cloth_scaled, y_test, False)
    }
    
    ml_data = {
        "train_x": ml_train_x, "train_y": y_train,
        "val_x": ml_val_x, "val_y": y_val,
        "test_x": ml_test_x, "test_y": y_test
    }
    
    return loaders, ml_data, x_train_body_scaled.shape[1], x_train_cloth_scaled.shape[1]
