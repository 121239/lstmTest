import pprint

from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams


# pytdx API文档 https://pytdx-docs.readthedocs.io/zh-cn/latest/pytdx_exhq/
api = TdxHq_API()
with api.connect('60.191.117.167', 7709):
    # data1 = api.get_company_info_content(0, '000001', '000001.txt', 0, 100)
    # print(data1)
    # 查询历史分时行情
    # data = api.get_history_minute_time_data(0, '000001', 20161209)
    # print(data) #OrderedDict({'price': 11.58, 'vol': 66418}) price价格  vol成交量
    # pprint.pprint(api.to_df(api.get_history_minute_time_data(0, '000001', 20161209)))
    # pprint.pprint(api.to_df(api.get_transaction_data(0, '000001', 0, 30)))
    # 查询历史分笔成交
    num = 2000
    i = 0
    datas= []
    # while num == 2000 :
    #     data = api.get_history_transaction_data(1, '000001', i*num, 2000,20250604)
    #     i = i + 1
    #     num = len(data)
    #     datas = data + datas
    #     # pprint.pprint(api.to_df(data))
    #
    # # pprint.pprint(api.to_df(datas))
    # # print(api.to_df(datas))
    # grouped = api.to_df(datas).groupby(['time', 'price','buyorsell'], as_index=False)['vol'].sum()
    # print(grouped)
    data = api.to_df(api.get_history_transaction_data(0, '000001', 0, 2000,20250603))
    pprint.pprint(data)
    # 明天看看实时的是怎么弄的 分笔成交的 应该是最近的 多拿点 然后过滤掉不要的数据就好了