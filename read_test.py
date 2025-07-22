import pandas as pd
import chardet
# # 如果是文本格式（如CSV）
# df = pd.read_csv('20250616.dat', sep='\t')  # 根据实际分隔符调整
#
# # 如果是二进制格式（需知道数据结构）
# print(df.head())


# with open('20250616.dat', 'r',encoding='ISO-8859-1') as file:
#     data = file.read()
#     print(data)
#
# with open('20250616.dat', 'r',encoding='utf-8', errors='ignore') as file:
#     data = file.read()
#     print(data)

# with open('20250616.dat', 'r', encoding='gbk', errors='ignore') as file:
#     data = file.read()
#     print(data)

# with open('20250616.dat', 'rb') as file:
#     data = file.read()
#     print(data)  # 这将打印出原始的二进制数据

with open('20250616.dat', 'rb') as file:
    raw_data = file.read(1000)  # 读取文件前10000个字节
    result = chardet.detect(raw_data)
    encoding = result['encoding']

print(f"文件的编码方式是: {encoding}")

with open('20250616.dat', 'r', encoding='windows-1252',errors='ignore') as file:
    data = file.read()
    print(data)

# with open('20250616.dat', 'r', encoding='latin1',errors='ignore') as file:
#     data = file.read()
#     print(data)
# with open('20250616.dat', 'rb') as file:
#     byte_data = file.read()
#
# # 按字节逐个打印，分析哪些字节导致问题
# # for byte in byte_data:
# #     print(hex(byte))
# decoded_data = byte_data.decode('gbk')
# print(decoded_data)
