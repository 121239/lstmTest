import os
import warnings
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import lightning.pytorch as pl  # 新版 PyTorch Lightning (>=2.0)
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet

warnings.filterwarnings("ignore")  # avoid printing out absolute paths
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from pytorch_forecasting.data.examples import get_stallion_data

data = get_stallion_data()

# add time index
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month  #2020年1月 → 2020×12 + 1 = 24241
data["time_idx"] -= data["time_idx"].min()          # 24241会变成0   24242会变成1

# add additional features
data["month"] = data.date.dt.month.astype(str).astype("category")  # 提取日期中的月份，并转换为分类变量（category）。
data["log_volume"] = np.log(data.volume + 1e-8) #取对数变换  + 1e-8：避免 volume=0 时出现 -∞（加一个极小值保证数值稳定）。  如果 volume 的分布右偏（大量小值，少数极大值），对数变换可以使其更接近正态分布，提升模型效果
data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"], observed=True).volume.transform("mean") # 计算每个时间点（time_idx）下每个 SKU（商品）的平均销量，并作为新特征。
data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"], observed=True).volume.transform("mean") #计算每个时间点（time_idx）下每个 agency（机构/代理商）的平均销量

# we want to encode special days as one variable and thus need to first reverse one-hot encoding
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]
data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category") #处理 special_days（特殊日期/事件）相关的列
print(data.sample(10, random_state=521)) #伪随机  	固定随机种子抽取10行


# 2

max_prediction_length = 6
max_encoder_length = 24
training_cutoff = data["time_idx"].max() - max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],  #筛选训练集数据，training_cutoff 是一个时间截断点（例如某个月份），用于划分训练集和验证集。
    time_idx="time_idx",#时间索引 指定数据中表示时间顺序的列（之前生成的连续整数索引）
    target="volume",#要预测的目标列（如销量 volume）
    group_ids=["agency", "sku"], #分组变量 定义时间序列的分组（例如，每个 agency 和 sku 组合是一个独立的时间序列）
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,## 编码器最大长度（历史窗口）
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,# 预测最大长度（未来窗口）
    static_categoricals=["agency", "sku"], #静态分类特征（不随时间变化） ?? static_categoricals：离散的分组特征（如机构 ID、商品类别）
    static_reals=["avg_population_2017", "avg_yearly_household_income_2017"], ## 静态数值特征 离散的分组特征（如机构 ID、商品类别）
    time_varying_known_categoricals=["special_days", "month"],# 已知的未来分类特征 time_varying_known_*：在预测时已知的特征（如日期、节假日、定价）。
    variable_groups={"special_days": special_days},  # 分类特征的分组 variable_groups：将多个分类变量（如 special_days 列表中的节日）合并为一组。
    time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"], # 已知的未来数值特征
    time_varying_unknown_categoricals=[], # 未知的分类特征（通常为空）
    time_varying_unknown_reals=[  # 未知的数值特征
        "volume",
        "log_volume",
        "industry_volume",
        "soda_volume",
        "avg_max_temp",
        "avg_volume_by_agency",
        "avg_volume_by_sku",
    ],
    target_normalizer=GroupNormalizer(
        groups=["agency", "sku"], transformation="softplus"
    ),  # use softplus and normalize by group  按分组（agency 和 sku）对目标变量 volume 进行归一化。 使用 Softplus 函数（保证输出为正数，适合销量等非负目标）
    add_relative_time_idx=True,# 添加相对时间索引
    add_target_scales=True,# 添加目标变量的缩放信息
    add_encoder_length=True, # 添加编码器长度信息
)

# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series  创建验证集，用于进行模型的评估
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)


# 3
#创建数据加载器
batch_size = 128  # set this between 32 to 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)


#
# 4
# 配置模型和训练器
pl.seed_everything(42) # 固定所有随机种子，确保实验可复现性  深度学习训练中的权重初始化、数据打乱（shuffle）、Dropout 等操作依赖随机数，固定种子可确保每次运行结果一致
# L.seed_everything(42) # 固定所有随机种子，确保实验可复现性  深度学习训练中的权重初始化、数据打乱（shuffle）、Dropout 等操作依赖随机数，固定种子可确保每次运行结果一致
trainer = pl.Trainer(
    accelerator="auto",  # 自动检测可用硬件（GPU/CPU）
    devices="auto",      # 自动选择设备数量
    gradient_clip_val=0.1,
)

#首要调参：
# hidden_size（模型容量）和 learning_rate（收敛速度）。
# 次要调参：
# dropout（过拟合时增加）、attention_head_size（大数据集增加）。
# 高级调参：
# hidden_continuous_size（影响连续变量处理）。

#   根据数据集自动推断输入特征（如静态变量、时间相关变量等），并初始化 TFT 模型。TFT 是一种专为时间序列设计的高性能模型，
tft = TemporalFusionTransformer.from_dataset(
    training,
    # 控制模型参数更新的步长。 初始值通常设为 0.01~0.1，可通过学习率扫描（lr_find()）调整 较大的学习率（如 0.03）适合快速收敛，但可能不稳定。
    learning_rate=0.03,
    hidden_size=16,  # 控制模型所有隐藏层的维度（如 LSTM、注意力层的神经元数）。 越大模型越强，但计算量越大 一般从 16~64 开始调优。
    # 多头注意力的头数。小数据集设为 1，大数据集可增至 4  ?? 多头不是可以用来控制特征的吗  错 不是直接增减特征数量，而是让模型从同一组输入特征中提取多角度的信息
    # 若 hidden_size=16 且 attention_head_size=4，则每个头处理 16 // 4 = 4 维的子空间。
    attention_head_size=1,
    dropout=0.1,  # 随机丢弃神经元比例，防止过拟合。常用 0.1~0.3。
    hidden_continuous_size=8,  # 处理连续变量的隐藏层维度，需 <= hidden_size。
    output_size=7,  # 预测7个分位数 ？？
    loss=QuantileLoss(),#分位数回归损失，直接优化预测区间（而不仅是均值）
    # 训练优化 若验证损失在 4 个 epoch 内未下降，自动降低学习率 避免模型陷入局部最优
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")



# 使用 Tuner 进行学习率查找
tuner = Tuner(trainer)
# 自动搜索 最佳初始学习率
# 避免手动猜测学习率（如 0.001 或 0.01 这类经验值）。
# 解决训练中因学习率不当导致的梯度爆炸（Loss → NaN）或收敛过慢问题。

res = tuner.lr_find(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(suggest=True, show=False)  # 关闭自动显示
fig.savefig("plot1.png")  # 保存为图片
# plt.show()  # ✅ 正确方式



# 这段代码是使用 PyTorch Lightning 框架时常见的回调函数（Callbacks）设置，主要用于模型训练过程中的优化和监控
# 作用：当验证损失(val_loss)在10个epoch内没有至少改善0.0001时，自动停止训练
# 防止模型过拟合，节省计算资源
early_stop_callback = EarlyStopping(
    monitor="val_loss",  # 监控验证集损失
    min_delta=1e-4,     # 认为有改进的最小变化量
    patience=10,        # 在停止前等待的epoch数
    verbose=False,      # 不打印详细消息
    mode="min"          # 目标是最小化监控指标
)
lr_logger = LearningRateMonitor()  # 记录学习率
logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

trainer = pl.Trainer(
    max_epochs=30,                  # 最多训练 30 个 epoch
    accelerator="auto",             # 自动检测 GPU/CPU
    devices="auto",                 # 自动选择可用设备（如多块 GPU）
    enable_model_summary=True,      # 打印模型结构摘要      enable_progress_bar=True,  # todo 显示进度条 enable_model_summary=True,  # 显示模型摘要
    gradient_clip_val=0.1,          # 梯度裁剪，防止梯度爆炸
    val_check_interval=30,          # 每 30 batch 验证
    check_val_every_n_epoch=None,   # 禁用 epoch 验证
    callbacks=[lr_logger, early_stop_callback],  # 学习率监控 + 早停
    logger=logger,                  # 使用 TensorBoard 记录训练日志
)
tft = TemporalFusionTransformer.from_dataset(
    training,                      # 训练数据集（包含特征信息）
    learning_rate=0.03,           # 初始学习率
    hidden_size=16,               # 隐层维度
    attention_head_size=1,         # 注意力头数
    dropout=0.1,                  # Dropout 比例
    hidden_continuous_size=8,     # 连续变量的隐层维度
    output_size=7,                # 输出分位数（默认 7 个）
    loss=QuantileLoss(),          # 分位数损失函数
    log_interval=10,              # 每 10 个 batch 记录一次日志
    reduce_on_plateau_patience=4,  # 验证损失 4 次未改善后降低学习率
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")



# fit network
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)


# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
# calcualte mean absolute error on validation set
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
(actuals - predictions).abs().mean()



# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
# raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
raw_output = best_tft.predict(val_dataloader, mode="raw", return_x=True)
x = raw_output.x  # 通过属性访问输入数据
raw_predictions = raw_output.output  # 获取原始预测字典
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True);




    # calcualte metric by which to display
predictions = best_tft.predict(val_dataloader)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)
indices = mean_losses.argsort(descending=True)  # sort losses
for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(
        x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
    )
    # plt.show()  # 确保图表显示


interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
best_tft.plot_interpretation(interpretation)


dependency = best_tft.predict_dependency(
    val_dataloader.dataset, "discount_in_percent", np.linspace(0, 30, 30), show_progress_bar=True, mode="dataframe"
)

# plotting median and 25% and 75% percentile
agg_dependency = dependency.groupby("discount_in_percent").normalized_prediction.agg(
    median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
)
ax = agg_dependency.plot(y="median")
ax.fill_between(agg_dependency.index, agg_dependency.q25, agg_dependency.q75, alpha=0.3);

plt.show()
print('11111111111111')
