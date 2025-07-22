# 代码详细解释

这段代码实现了一个基于Temporal Fusion Transformer (TFT)的时间序列预测模型，用于预测商品销量。以下是逐步解析：

## 1. 数据准备与预处理

python

```
data = get_stallion_data()  # 获取示例数据
data["time_idx"] = data["date"].dt.year * 12 + data["date"].dt.month  # 创建时间索引
data["time_idx"] -= data["time_idx"].min()  # 标准化时间索引从0开始

# 特征工程
data["month"] = data.date.dt.month.astype(str).astype("category")  # 月份作为分类变量
data["log_volume"] = np.log(data.volume + 1e-8)  # 对数变换销量
data["avg_volume_by_sku"] = data.groupby(["time_idx", "sku"]).volume.transform("mean")
data["avg_volume_by_agency"] = data.groupby(["time_idx", "agency"]).volume.transform("mean")

# 特殊日期处理
special_days = [...]  # 特殊日期列表
data[special_days] = data[special_days].apply(lambda x: x.map({0: "-", 1: x.name})).astype("category")
```

## 2. 创建时间序列数据集

python

```
max_prediction_length = 6  # 预测未来6个月
max_encoder_length = 24  # 使用过去24个月作为历史
training_cutoff = data["time_idx"].max() - max_prediction_length  # 训练集截止点

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="volume",
    group_ids=["agency", "sku"],  # 按代理商和商品分组
    # 各种长度和特征设置...
    target_normalizer=GroupNormalizer(groups=["agency", "sku"], transformation="softplus")
)

validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
```

## 3. 数据加载器

python

```
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
```

## 4. 模型配置与训练

python

```
# 初始化TFT模型
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7个分位数
    loss=QuantileLoss(),
    reduce_on_plateau_patience=4
)

# 学习率查找
tuner = Tuner(trainer)
res = tuner.lr_find(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

# 训练配置
trainer = pl.Trainer(
    max_epochs=30,
    callbacks=[EarlyStopping(patience=10), LearningRateMonitor()],
    logger=TensorBoardLogger("lightning_logs")
)

# 训练模型
trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
```

## 5. 模型评估与解释

python

```
# 加载最佳模型
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# 计算验证集上的平均绝对误差
actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
predictions = best_tft.predict(val_dataloader)
mae = (actuals - predictions).abs().mean()

# 可视化预测结果
raw_predictions, x = best_tft.predict(val_dataloader, mode="raw", return_x=True)
best_tft.plot_prediction(x, raw_predictions, idx=0)

# 模型解释
interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
best_tft.plot_interpretation(interpretation)
```

# 代码总结

这段代码实现了一个完整的时间序列预测流程：

1. **数据准备**：加载示例数据，进行时间索引创建和特征工程，包括对数变换、分组统计特征和特殊日期处理。
2. **数据集创建**：使用TimeSeriesDataSet构建适合TFT模型的时间序列数据集，明确划分编码器长度(历史窗口)和预测长度(未来窗口)。
3. **模型配置**：初始化Temporal Fusion Transformer模型，关键参数包括：
    - hidden_size=16 (模型容量)
    - learning_rate=0.03 (学习率)
    - attention_head_size=1 (注意力头数)
    - output_size=7 (预测7个分位数)
4. **训练过程**：
    - 使用学习率查找器确定最佳初始学习率
    - 配置早停机制(EarlyStopping)防止过拟合
    - 使用TensorBoard记录训练日志
    - 训练30个epoch或在验证损失10次未改善时停止
5. **评估与解释**：
    - 加载验证集上表现最好的模型
    - 计算平均绝对误差(MAE)
    - 可视化预测结果
    - 分析模型对各个特征的依赖关系

该解决方案特别适合具有以下特点的时间序列预测问题：

- 多组相关时间序列(如不同代理商的不同商品)
- 需要同时利用历史模式和已知未来信息(如定价、促销)
- 需要预测不确定性(通过分位数预测)

关键优势在于TFT模型能够：

1. 自动学习时间依赖关系
2. 处理静态特征和时变特征
3. 提供可解释的特征重要性
4. 输出预测区间而不仅是点估计