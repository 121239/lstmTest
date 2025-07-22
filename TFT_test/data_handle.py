import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import os

# 读取数据并保存
def data_handle(name):
    print(name)
    # 读取数据
    df = pd.read_csv(
        path + name+".xlsx",
        sep="\t",  # 制表符分隔
        skiprows=1,  # 跳过第1行（标题说明行）
        skipfooter=1,  # 跳过最后1行
        engine="python",  # 必须指定engine='python'才能使用skipfooter
        encoding="gbk"  # 中文编码
    )


    df.columns = df.columns.str.strip()  # 将 '      日期' → '日期'
    print(df.columns)  # 检查列名是否修正

    # 1. 规范时间格式
    df['时间'] = df['时间'].astype(str).str.zfill(4)  # 补零到4位
    df['时间'] = df['时间'].str[:2] + ':' + df['时间'].str[2:4] + ':00'

    # 2. 合并日期时间
    df['datetime'] = pd.to_datetime(df['日期']) + pd.to_timedelta(df['时间'])
    # 添加交易时段标记
    # df['交易时段'] = df['datetime'].dt.strftime('%H:%M') + '-' + \
    #                  (df['datetime'] + pd.Timedelta(hours=1)).dt.strftime('%H:%M')

    # 定义交易时间段范围
    trading_hours = [
        ('09:30:00', '10:30:00'),  # 必须包含秒数
        ('10:30:00', '11:30:00'),
        ('13:00:00', '14:00:00'),
        ('14:00:00', '15:00:00')
    ]
    # 将时间字符串转换为时间对象（仅时间部分）
    trading_periods = [(pd.to_datetime(start).time(), pd.to_datetime(end).time())
                       for start, end in trading_hours]


    # 创建一个函数，将时间戳映射到对应的交易时段
    def map_to_trading_period(dt):
        dt_time = dt.time()
        for i, (start, end) in enumerate(trading_periods):
            if start <= dt_time <= end:
                return i+1  # 返回时段索引（也可返回自定义标签）
        return None  # 非交易时段返回None

    # 处理数据
    result = (
        df.set_index('datetime')
        # 为每个时间戳分配对应的交易时段
        .assign(period=lambda x: x.index.map(map_to_trading_period))
        .assign(weekday=lambda x: x.index.day_name())
        # 按交易时段分组
        .groupby(['日期','period'], dropna=False)
        # 聚合操作
        .agg({
            'weekday': 'first',
            '开盘': 'first',
            '最高': 'max',
            '最低': 'min',
            '收盘': 'last',
            '成交量': 'sum',
            '成交额': 'sum'
        })
        # 删除非交易时段的数据（period为None的行）
        .dropna(subset=['开盘'], how='any')
        # 重置索引（可选）
        .reset_index()

    )
    # 保存为新的Excel文件
    output_path = path +"OneHour/"+ name+".csv"

    # 删除旧文件（如果存在）
    if os.path.exists(output_path):
        os.remove(output_path)

    # 保存为新 CSV
    result.to_csv(output_path, index=False, encoding='utf-8-sig')




def process_xlsx_files(folder_path):
    """处理文件夹中所有xlsx文件"""
    # 获取文件夹中所有.xlsx文件
    xlsx_files = [f.replace('.xlsx', '') for f in os.listdir(folder_path)
                  if f.endswith('.xlsx') and os.path.isfile(os.path.join(folder_path, f))]

    if not xlsx_files:
        print(f"文件夹 {folder_path} 中没有找到.xlsx文件")
        return

    print(f"找到 {len(xlsx_files)} 个.xlsx文件:")
    for i, filename in enumerate(xlsx_files, 1):
        print(f"{i}. {filename}")

    # xlsx_files = xlsx_files[xlsx_files.index('SZ#000012'):]

    # 逐个处理文件
    # for filename in xlsx_files:
    #     data_handle(filename)  # 调用处理函数
    #  多线处理
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(data_handle, xlsx_files)



path = "D:/MyTool/tdx/T0002/export/"
if __name__ == "__main__":

    # data_handle('SZ#000012')
    # 确保文件夹存在
    if not os.path.exists(path):
        print(f"错误: 文件夹 {path} 不存在")
    else:
        process_xlsx_files(path)