from xtquant import xtdata
# 订单流数据仅提供1m周期数据下载，其他周期的订单流数据都是通过1m周期合成的
period = "orderflow1m"
# 下载000001.SZ的1m订单流数据
# xtdata.download_history_data("000001.SZ",period=period)
# # 获取000001.SZ的1m订单流数据
data = xtdata.get_market_data_ex([],["000001.SZ"],period=period)["000001.SZ"]
# print(data)

print(xtdata.get_full_tick(['000001.SZ']))
# stocks = xtdata.get_stock_list_in_sector('沪深A股')
# print(stocks)