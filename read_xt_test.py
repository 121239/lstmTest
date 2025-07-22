from xtquant import xtdata
import pandas as pd
import datetime,time
code =['000001.SZ']
period = 'orderflow1m'
day = '20250620'
data_dir=r'/'
kline_data=xtdata.get_local_data(field_list=[],stock_list=["000001.SZ"],period=period, start_time=day,end_time=day,count=10,data_dir=data_dir)
print(kline_data)
# df =pd.concat([kline_data[i].T for i in ['time', 'open', 'high','low','close','volume','amount','settelementPrice', 'openInterest']],axis=1)
# df.columns=['time','open','high','low','close','volume','amount','settelementPrice','openInterest']
# df['time']=df['time'].apply(lambda x: datetime.datetime.fromtimestamp(x/1000.0))
# # print(tabulate(df,headers=df.columns,))
# print(df.shape[0])