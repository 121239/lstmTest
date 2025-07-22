import pprint

from pytdx.hq import TdxHq_API
from pytdx.exhq import TdxExHq_API
from pytdx.params import TDXParams


# pytdx API文档 https://pytdx-docs.readthedocs.io/zh-cn/latest/pytdx_exhq/
api = TdxExHq_API()
with api.connect('175.24.47.69', 7727):  # 连接不上 查 netstat -ano | findstr "TDX的PID"
# with api.connect('61.152.107.141', 7727):  # 连接不上 查 netstat -ano | findstr "11436"
    # 获取市场代码
    data = api.get_markets()
    print(data)
    #
    # data1 = api.get_instrument_quote(47, "IF1709")
    # print(data1)
    #
    # data1 = api.get_history_minute_time_data(31, "00020", 20170811)
    # print(data1)

    # pprint.pprint(api.to_df(api.get_instrument_info(0, 100)))
    pprint.pprint(api.to_df(api.get_history_transaction_data(31, "00020", 20250603)))
    #
    # print(f"买一价: {data['buy_one']}")
    # print(f"买一量: {data['buy_one_volume']}")
    # print(f"卖一价: {data['sell_one']}")
    # print(f"卖一量: {data['sell_one_volume']}")