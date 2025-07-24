import os
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl  # 新版 PyTorch Lightning (>=2.0)
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
def load_and_prepare_data(stock_files):
    """
    加载多只股票数据并预处理
    stock_files: 股票数据文件路径列表
    """
    all_data = []

    for i, file_path in enumerate(stock_files):
        # 加载单只股票数据
        print(file_path)
        df = pd.read_csv(path+file_path)
        if(len(df) < 360) :
            continue

        # if i > 500:
        #     continue

        # 添加股票ID
        stock_id = os.path.basename(file_path).split('.')[0]
        df['stock_id'] = stock_id
        df['group_id'] = i  # 数字ID用于分组

        # 转换日期格式
        df['date'] = pd.to_datetime(df['日期'])
        df = df.sort_values(['date', 'period'])

        # 创建全局时间索引 (每只股票独立计时)
        df['time_idx'] = df.groupby('stock_id').cumcount()

        # 添加时间特征
        df['hour'] = df['period'] - 1
        df['hour'] = df['hour'].astype(str)  # 转换为字符串
        df['weekday'] = df['weekday'].astype(str)  # 转为字符串
        df['month'] = df['date'].dt.month
        df['weekday'] = df['weekday'].apply(
            lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(x)
        )

        # 对数值特征进行标准化  每只股票独立进行标准化    todo  使用对数收益代替价格 对预测比较有帮助
        numeric_cols = ['开盘', '最高', '最低', '收盘']
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        # for col in numeric_cols:
        #     df[col] = df[col].transform(
        #         lambda x: (x - x.mean()) / (x.std() + eps) # 避免除零  #标准化 将数据转换为均值为0，标准差为1的分布（即z-score标准化）  mean() 平均值 std 标准差
        #     )

        all_data.append(df)


    # 合并所有股票数据
    all_data = [df for df in all_data if not df.empty]  # 过滤空 DataFrame
    full_df = pd.concat(all_data, ignore_index=True)
    # 换成非中文的名称
    column_mapping = {
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close'
    }
    # 使用rename方法重命名列
    full_df = full_df.rename(columns=column_mapping)

    # 特征工程增强
    # 按股票标准化数值特征
    # 添加滞后特征 (1,2,3,4,8,12小时)
    # 添加移动平均线 (MA_6) 和价格变化率
    # 填充缺失值确保数据连续性

    # 添加滞后特征
    for lag in [1, 2, 3, 4, 8, 12]:  # 1-12小时滞后  todo 这是啥 问一下
        full_df[f'close_lag_{lag}'] = full_df.groupby('stock_id')['close'].shift(lag)

    # 添加技术指标
    full_df['MA_6'] = full_df.groupby('stock_id')['close'].transform(lambda x: x.rolling(6).mean())
    # full_df['price_change'] = full_df.groupby('stock_id')['收盘'].pct_change()
    full_df['price_change'] = full_df.groupby('stock_id')['close'].transform(
        lambda x: np.log((x + eps) / (x.shift(1) + eps))
    ).replace([np.inf, -np.inf], np.nan).fillna(0)



    # 删除列  太大了 先删除
    full_df = full_df.drop(['成交量','成交额'],axis=1)

    # 填充缺失值
    full_df = full_df.bfill()

    # 删除旧文件（如果存在）
    if os.path.exists(output_path):
        os.remove(output_path)

    full_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    return full_df

# 2. 创建数据集
def create_datasets(df, prediction_horizon = 8, history_length=24):
    """
    创建TFT数据集
    prediction_horizon: 预测时间步长 (8小时=2天)
    history_length: 历史数据长度 (24小时=6天)
    """
    # 训练/验证集分割点 (按时间索引)
    max_time = df["time_idx"].max()
    training_cutoff = max_time - prediction_horizon - history_length

    # 分类特征处理
    categorical_cols = ["hour", "weekday", "month"]
    for col in categorical_cols:
        df[col] = df[col].astype(str).astype("category")

        # 创建数据集时添加过滤条件，减少内存占用
    train_filter = lambda x: x.time_idx <= training_cutoff
    val_filter = lambda x: (x.time_idx > training_cutoff - history_length) & (x.time_idx <= max_time)

    # 使用更高效的数据类型
    real_cols = ["open", "high", "low", "close_lag_1", "close_lag_2",
                 "close_lag_3", "close_lag_4", "close_lag_8",
                 "close_lag_12", "MA_6", "price_change"]
    for col in real_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")
    # 创建时间序列数据集
    dataset = TimeSeriesDataSet(
        df[train_filter],
        time_idx="time_idx",
        target="close",
        group_ids=["group_id"],
        min_encoder_length=history_length // 2,
        max_encoder_length=history_length,
        min_prediction_length=1,
        max_prediction_length=prediction_horizon,
        static_categoricals=["stock_id"],
        time_varying_known_categoricals=categorical_cols,#已知的未来分类特征
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=real_cols,
        target_normalizer=GroupNormalizer(groups=["group_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # 创建验证集
    validation = TimeSeriesDataSet.from_dataset(
        dataset,
        df[val_filter],
        predict=True,
        stop_randomization=True
    )

    return dataset, validation

# 3. 训练TFT模型
def train_tft_model(train_dataset, val_dataset, epochs=100):

    # 自动确定最优批大小
    def find_optimal_batch_size():
        max_batch_size = 256
        min_batch_size = 32
        current_batch = min_batch_size

        while current_batch <= max_batch_size:
            try:
                # 测试当前批大小是否会导致OOM
                test_loader = train_dataset.to_dataloader(
                    train=True,
                    batch_size=current_batch,
                    num_workers=min(8, os.cpu_count()//2),
                    pin_memory=torch.cuda.is_available()
                )
                for batch in test_loader:
                    pass
                return current_batch
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    current_batch = current_batch // 2
                    print(f"批大小 {current_batch*2} 导致OOM，尝试 {current_batch}")
                    break
                else:
                    raise
        return max(min_batch_size, current_batch)


    batch_size = find_optimal_batch_size()
    print(f"自动确定的批大小: {batch_size}")

    # 数据加载器配置
    train_dataloader = train_dataset.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=min(16, os.cpu_count()-1),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True  # 减少重复初始化开销
    )
    val_dataloader = val_dataset.to_dataloader(
        train=False,
        batch_size=batch_size*2,  # 验证集可用更大批次
        num_workers=min(16, os.cpu_count()-1),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True
    )

    # 模型配置
    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=0.03,  # 更保守的初始值
        hidden_size=32,
        attention_head_size=4,
        dropout=0.15,  # 稍高的dropout防止过拟合
        hidden_continuous_size=16,
        output_size=7,  # 7个分位数
        loss=QuantileLoss(quantiles=[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 回调函数
    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=12,  # 更长的耐心
        verbose=True,
        mode="min",
        check_finite=True  # 检查NaN值
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 训练器配置
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices="auto",
        gradient_clip_val=0.2,  # 稍宽松的梯度裁剪
        callbacks=[early_stop, lr_monitor],
        enable_checkpointing=True,
        default_root_dir="tft_logs",
        deterministic=True,  # 增强可复现性
        precision="16-mixed" if torch.cuda.is_available() else "32-true",  # 自动混合精度
        accumulate_grad_batches=2 if batch_size < 64 else 1,  # 小批次时梯度累积
        logger=True,
        enable_progress_bar=True,
        overfit_batches=0  # 禁用过拟合检测
    )
    # 学习率查找 (安全模式)
    try:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            min_lr=1e-6,
            max_lr=0.3,
            num_training=100,
            early_stop_threshold=None  # 禁用自动停止
        )
        suggested_lr = lr_finder.suggestion()
        print(f"建议学习率: {suggested_lr:.5f}")
        tft.learning_rate = suggested_lr
    except Exception as e:
        print(f"学习率查找失败: {str(e)}，使用默认值0.03")
        tft.learning_rate = 0.03

    # 完整训练
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 保存最佳模型
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"最佳模型保存于: {best_model_path}")

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return tft

# 4. 预测函数
def predict_future(tft, dataset, stock_ids, prediction_horizon=8):
    """
    预测未来股价
    stock_ids: 要预测的股票ID列表
    """
    all_predictions = []

    for stock_id in stock_ids:
        # 获取该股票的最新数据
        stock_data = dataset[dataset['stock_id'] == stock_id]
        last_data = stock_data[stock_data.time_idx > stock_data.time_idx.max() - dataset.max_encoder_length]

        # 生成预测
        raw_predictions, x = tft.predict(
            last_data,
            mode="raw",
            return_x=True,
            n_samples=200  # 蒙特卡洛采样
        )

        # 提取预测结果（中位数）
        median_prediction = raw_predictions.output.prediction[..., 3].median(dim=1).values.cpu().numpy()[0]

        # 创建预测时间索引
        last_date = stock_data['date'].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(hours=1),
            periods=prediction_horizon,
            freq="H"
        )

        # 保存结果
        stock_pred = pd.DataFrame({
            "stock_id": stock_id,
            "timestamp": future_dates,
            "predicted_close": median_prediction
        })
        all_predictions.append(stock_pred)

    return pd.concat(all_predictions)

def process_xlsx_files(folder_path):
    """处理文件夹中所有csv文件"""
    # 获取文件夹中所有.csv文件
    xlsx_files = [f.replace('', '') for f in os.listdir(folder_path)
                  if f.endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))]

    if not xlsx_files:
        print(f"文件夹 {folder_path} 中没有找到.csv文件")
        return

    print(f"找到 {len(xlsx_files)} 个.csv文件:")
    for i, filename in enumerate(xlsx_files, 1):
        print(f"{i}. {filename}")

    return xlsx_files

path = "D:/MyTool/tdx/T0002/export/OneHour/"
output_path = path +"dataHandle/20250723.csv"
# output_path = "20250723.csv"
eps = 1e-6  # 极小值，防止除零
def check(full_data):
    # numeric_cols = full_data.columns  # 替换为你的实际列名
    # print("检查无穷大 (inf/-inf):")
    # print(full_data[numeric_cols].isin([np.inf, -np.inf]).sum())
    #
    # print("\n检查 NaN:")
    # print(full_data[numeric_cols].isna().sum())

    # 检查每个stock_id的时间点数量
    time_counts = full_data.groupby('stock_id')['time_idx'].nunique()
    print("Time points per stock:\n", time_counts.value_counts())
    print(full_data['close'].isna().sum())  # 检查 NaN 的数量
    print((full_data['close'] == np.inf).sum())  # 检查 Inf 的数量
    print((full_data['close'] == -np.inf).sum())  # 检查 -Inf 的数量
    # 检查数值范围是否合理
    print(full_data['close'].describe())

    # 使用PyTorch严格检查
    tensor = torch.tensor(full_data['close'].values, dtype=torch.float32)
    print(f"非有限值数量: {(~torch.isfinite(tensor)).sum().item()}")
    exit()

# 主执行流程  todo 模型解释
if __name__ == "__main__":


    # 加载多只股票数据
    # stock_files = process_xlsx_files(path)
    # # # 预处理数据
    # full_data = load_and_prepare_data(stock_files)
    # exit()

    full_data =  pd.read_csv(output_path)
    # check(full_data)
    print("GPU available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    # 创建数据集
    prediction_horizon = 8  # 预测未来8小时 (2天)
    history_length = 24      # 使用24小时历史数据
    train_dataset, val_dataset = create_datasets(full_data, prediction_horizon, history_length)

    # 训练模型
    tft_model = train_tft_model(train_dataset, val_dataset, epochs=100)

    # 保存模型
    torch.save(tft_model.state_dict(), "multi_stock_tft.pth")

    # 预测特定股票
    stocks_to_predict = ["stock_000001", "stock_600000"]
    predictions = predict_future(tft_model, full_data, stocks_to_predict, prediction_horizon)

    # 输出预测结果
    print("未来2天股价预测:")
    print(predictions.pivot(index="timestamp", columns="stock_id", values="predicted_close"))

    # 可视化预测结果
    predictions.set_index("timestamp").groupby("stock_id")["predicted_close"].plot(
        title="2-Day Stock Price Prediction"
    )