import os
import time

import pyautogui
import win32gui
import win32con
from pywinauto import Application

import tdx_win_tick_down




def get_stock_code(stock_code):
    if str(stock_code)[0] in ('0','3') :    # 深圳股票
        return '6'+stock_code
    elif str(stock_code)[0] == '6' : return '7'+stock_code
    else: return '4'+stock_code


def tdx_post_message(stock_code):
    # 取消多余窗口
    try:
        app = Application(backend="win32").connect(title_re=".*PageUp/Down:前后日 空格键:操作*")  # 匹配含的窗口
        tdx_window = app.window(title_re=".*PageUp/Down:前后日 空格键:操作*")
        # 获取控件在屏幕上的坐标 (left, top, right, bottom)
        rect = tdx_window.rectangle()
        center_x = (rect.left + rect.right) // 2
        center_y = (rect.top + rect.bottom) // 2
        tdx_window.double_click_input(coords=(center_x - rect.left, center_y - rect.top))
        pyautogui.press('esc')
        pyautogui.press('esc')
        pyautogui.press('esc')
        pyautogui.press('esc')
    except Exception as e:
        print(e)

    # 广播
    UWM_STOCK = win32gui.RegisterWindowMessage("Stock")
    # stock_code = '000001'
    code = int(get_stock_code(stock_code))
    print(code)
    win32gui.PostMessage(win32con.HWND_BROADCAST, UWM_STOCK, code, 0)
    time.sleep(2)

def save_data_time_list(stock_code='000031'):
    stock_files = tdx_win_tick_down.process_xlsx_files(stock_code)

    output_path = stock_code+'_data_list.txt'
    # 删除旧文件（如果存在）
    if os.path.exists(output_path):
        os.remove('output_path')
    # 使用 'w' 模式打开文件，如果文件不存在会创建一个新文件
    with open(output_path, 'w') as file:
        for item in stock_files:
            file.write(f"{item}\n")  # 每个元素占一行

def read_txt_file(file='000031_data_list.txt'):
    with open(file, 'r') as file:
        lines = file.readlines()  # 每行读取到一个列表
    my_list = [line.strip() for line in lines]  # 去除换行符
    return my_list


if __name__ == '__main__':
    data_list = read_txt_file()
    print(data_list)

    app = Application(backend="win32").connect(title_re=".*通达信专业研究版V7.*")  # 匹配含"通达信"的窗口
    print(len(app.windows()))
    # for w in app.windows():
    #     # print(f"窗口标题: {w.window_text()}, 类名: {w.class_name()},类名2:{w.class_name}")
    #     print(f"窗口标题: {w.window_text()}, 类名: {w.class_name()}")
    tdx_window = app.window(class_name="TdxW_MainFrame_Class")
    tdx_window.set_focus()  # 激活窗口

    type = 'SZ#'
    stock_list = tdx_win_tick_down.get_stock_List(type)
    print(f'{type} 数量 :{len(stock_list)}')
    # 报错超过20次 结束
    error_num = 0
    for stock_code in stock_list:

        try:
            # 已下载的数据
            stock_files = tdx_win_tick_down.process_xlsx_files(stock_code)
            #跳过 stock_files存在 并且有 data_list 的日期
            if stock_files and set(data_list).issubset(set(stock_files)):
                print(f'{stock_code} 已存在数据 {len(stock_files)} 跳过')
                continue

            tdx_post_message(stock_code)
            # 双击之后再确认 可以弹出分笔数据的框
            rect = tdx_window.rectangle()
            center_x = (rect.left + rect.right) // 2
            center_y = (rect.top + rect.bottom) // 4
            # print(center_x, center_y)
            print("开始执行"+stock_code)

            try:
                tdx_window.double_click_input(coords=(center_x - rect.left, center_y - rect.top))
                time.sleep(0.1)
                print(len(app.windows()))
                pyautogui.press('enter')
                app = Application(backend="win32").connect(title_re=".*PageUp/Down:前后日 空格键:操作*")
            except Exception as e:
                print(f"应该是分时页面，再次尝试",{str(e)})
                tdx_window.double_click_input(coords=(center_x - rect.left, center_y - rect.top))
                time.sleep(0.1)
                pyautogui.press('enter')
                tdx_window.double_click_input(coords=(center_x - rect.left, center_y - rect.top))
                time.sleep(0.1)
                pyautogui.press('enter')
                app = Application(backend="win32").connect(title_re=".*PageUp/Down:前后日 空格键:操作*")

            tdx_win_tick_down.get_tick_down(stock_code,stock_files)

        except Exception as e:
            error_num += 1
            if error_num > 20:
                raise e
            print(f'{stock_code} 报错 从试一下 num {error_num}')





