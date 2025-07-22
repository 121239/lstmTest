import win32com.client  # 引入Windows COM接口库
import time            # 引入时间模块

# 初始化通达信接口
tdx = win32com.client.Dispatch("79CEEA4E-C231-4614-9E3B-53B2A02F39B7")  # 根据版本选择适当的ID

# 获取实时数据
def get_realtime_data(stock_code):
    # 设定市场（0为沪市，1为深市）
    market = 0
    # 获取实时行情数据
    data = tdx.GetMktDataEx(market, stock_code)
    return data

# 主程序
if __name__ == "__main__":
    stock_code = "600519"  # 以贵州茅台为例
    while True:
        data = get_realtime_data(stock_code)
        print(f"股票代码: {stock_code} 当前价: {data['price']}")
        time.sleep(5)  # 每5秒获取一次数据