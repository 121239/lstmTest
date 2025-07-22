import pprint
from pytdx.hq import TdxHq_API

api = TdxHq_API()
with api.connect('124.71.187.122', 7709):
    print("连接成功")
    # 训练的编码
    stock_list= ['000001']
    start_date = 20221209
    end_date = 20241209
    for stock in stock_list:
        # 查询历史分时行情
        data = api.get_history_minute_time_data(0, stock, start_date)