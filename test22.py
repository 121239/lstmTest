import pandas as pd
import os
print(os.getcwd())  # 查看当前工作目录

data = pd.read_excel('D:\\920118.xlsx',engine='openpyxl')
# data = pd.read_csv('D:\\920118.csv',engine='xlrd')
data = data[['    收盘']]
print(data)
data = data['    收盘']
print(data)