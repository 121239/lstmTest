import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from keras.models import Sequential,load_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False

print("11111")

# 加载数据
data = pd.read_excel('D:\\501018_2.xlsx',engine='openpyxl')

# 假设你预测的列是 'value'，你可以根据需要修改列名
data = data[['    收盘']]

# 标准化数据   转换数据为0-1的数据
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 创建数据集，准备训练和测试数据
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])   # 取过去time_step个时间点的数据
        y.append(data[i + time_step, 0])       # 取下一个时间点的值作为标签
    return np.array(X), np.array(y)

time_step = 60  # 过去60天的数据预测明天的值
X, y = create_dataset(scaled_data, time_step)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 重塑数据，使其适配LSTM模型的输入要求  reshape(样本数, 时间步长, 特征数)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)



# 构建LSTM模型
model = Sequential()

# 添加LSTM层
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))

# 添加全连接层
model.add(Dense(units=1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练LSTM模型
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

model.save('my_test_model.keras')
# 加载模型
# model = load_model('my_test_model.keras')
# 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反标准化
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# 可视化结果
# 初始化与原始数据形状相同的空数组
plot_data = np.empty_like(data)
plot_data[:, :] = np.nan  # 全部填充为NaN

# 填充训练集预测结果
train_start = time_step  # 第一个预测点从time_step开始
train_end = train_start + len(train_predict)
plot_data[train_start:train_end, :] = train_predict

# 填充测试集预测结果
test_start = train_end + 1  # 测试集从训练集结束的下一个点开始
test_end = test_start + len(test_predict)
plot_data[test_start:test_end, :] = test_predict

# 绘制图表
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(scaled_data), label='真实值')
plt.plot(plot_data, label='预测值')  # 合并训练和测试预测结果
plt.legend()
# plt.show()
plt.savefig('output11.png')

