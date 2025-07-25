import os

import pyautogui
from pywinauto import Application
import time

from tdx_win_unit import extract_date, extract_code


def get_tick_down(stock_code):
    #下载tick数据

    app = Application(backend="win32").connect(title_re=".*PageUp/Down:前后日 空格键:操作*")  # 匹配含的窗口
    tdx_window = app.window(title_re=".*PageUp/Down:前后日 空格键:操作*")
    print(extract_date(tdx_window.window_text(), return_datetime=True))
    # tdx_window.set_focus()  # 激活窗口

    # 获取控件在屏幕上的坐标 (left, top, right, bottom)
    rect = tdx_window.rectangle()
    center_x = (rect.left + rect.right) // 2
    center_y = (rect.top + rect.bottom) // 2
    print(center_x, center_y)

    data_time = extract_date(tdx_window.window_text(), return_datetime=True)
    print(data_time)
    time.sleep(0.1)
    tdx_window.double_click_input(coords=(center_x - rect.left, center_y - rect.top))
    time.sleep(0.1)
    # 运行到最近的时间
    num = 50
    while num > 0:
        pyautogui.scroll(-1000)
        num -= 1

    print("回退一点")
    # 最近的时间有些没数据的 现在是测试 这些先不要
    while num < 5:
        pyautogui.scroll(1000)
        num += 1
    # 已下载的数据
    stock_files = process_xlsx_files(stock_code)
    for i in range(300):
        data_time = extract_date(tdx_window.window_text())
        if i % 5 == 0:
            print(data_time)

        data = data_time.replace('年', '').replace('月','').replace('日','')
        if data in stock_files:
            print(f'数据:{data_time}已存在 跳过')
            pyautogui.scroll(1000)
            time.sleep(0.1)
            continue

        # 点击控件中心
        tdx_window.right_click_input(coords=(center_x - rect.left, center_y - rect.top))  # 相对坐标
        time.sleep(0.5)
        if (i == 0):
            pyautogui.press('pageup')
            time.sleep(0.2)

        # 捕获弹出菜单（根据实际类名调整）
        popup_menu = app.window(class_name="#32768")
        if popup_menu.exists():
            # 键盘操作 导出分笔数据
            pyautogui.press('down', presses=13)  # 按↓键13次
            pyautogui.press('enter')  # 回车确认
            time.sleep(0.1)
            pyautogui.press('enter')
            time.sleep(0.1)
            if app.window(title="TdxW", class_name="#32770").exists():
                pyautogui.press('esc')  # 通用取消 取消弹框
            else:
                print("数据导出失败")
        else:
            print("菜单未弹出！")
            # continue

        tdx_window.click_input(coords=(center_x - rect.left, center_y - rect.top))
        pyautogui.press('pageup')
        time.sleep(0.2)

    # 取消多余窗口
    pyautogui.press('esc')
    pyautogui.press('esc')
    pyautogui.press('esc')
    pyautogui.press('esc')


def process_xlsx_files(stock_code):
    """处理文件夹中所有csv文件"""
    folder_path = 'd:/mytool/tdx/T0002/export2'
    # 获取文件夹中所有.csv文件
    xlsx_files = [f.replace('_' + stock_code + '.xls', '') for f in os.listdir(folder_path)
                  if f.endswith('_' + stock_code + '.xls') and os.path.isfile(os.path.join(folder_path, f))]

    if not xlsx_files:
        print(f"文件夹 {folder_path} 中没有找到.xls文件")
        return

    print(f"找到 {len(xlsx_files)} 个.xls文件:")
    # for i, filename in enumerate(xlsx_files, 1):
    #     print(f"{i}. {filename}")

    return xlsx_files

def get_stock_List(type='SZ#'):
    """处理文件夹中所有csv文件"""
    folder_path = 'd:/mytool/tdx/T0002/export_day'
    # 获取文件夹中所有.csv文件
    xlsx_files = [f.replace('.xlsx', '').replace(type,'') for f in os.listdir(folder_path)
                  if f.startswith(type) and f.endswith('.xlsx') and os.path.isfile(os.path.join(folder_path, f))]

    if not xlsx_files:
        print(f"文件夹 {folder_path} 中没有找到.xlsx文件")
        return

    print(f"找到 {len(xlsx_files)} 个.xls文件:")

    return xlsx_files