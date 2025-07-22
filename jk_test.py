from jqdatasdk import *

auth('19305698106', '123456@Xzl')

# stock_code = '000001.XSHE'  # 股票代码，例如：深证A股的平安银行
# start_date = '2023-01-01'   # 起始日期
# end_date = '2023-12-31'     # 结束日期
# frequency = 'daily'         # 日频数据，支持 daily, minute 等
#
# # 使用 get_price 函数获取数据
d = get_ticks("000001.XSHE",start_dt=None, end_dt="2025-03-03", count=8)
print(d)


#查询平安银行行情数据,round=False
# df =get_price('000001.XSHE', start_date= '2025-01-15 09:00:00',end_date='2025-01-17 14:00:00',fq='post', frequency='minute', fields=['open','close','low','high','volume','money','factor',
#                                                                                                                                      'high_limit','low_limit','avg','pre_close','paused','factor'],round=False)
# print(df[:4])
