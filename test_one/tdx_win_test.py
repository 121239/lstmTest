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

if __name__ == '__main__':

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
    for stock_code in stock_list:
        tdx_post_message(stock_code)
        # 双击之后再确认 可以弹出分笔数据的框
        rect = tdx_window.rectangle()
        center_x = (rect.left + rect.right) // 2
        center_y = (rect.top + rect.bottom) // 4
        # print(center_x, center_y)
        print("开始执行"+stock_code)
        try:
            tdx_window.double_click_input(coords=(center_x - rect.left, center_y - rect.top))
            time.sleep(0.2)
            print(len(app.windows()))
            pyautogui.press('enter')
            app = Application(backend="win32").connect(title_re=".*PageUp/Down:前后日 空格键:操作*")
        except Exception as e:
            print(f"应该是分时页面，再次尝试",{str(e)})
            tdx_window.double_click_input(coords=(center_x - rect.left, center_y - rect.top))
            time.sleep(0.2)
            pyautogui.press('enter')
            tdx_window.double_click_input(coords=(center_x - rect.left, center_y - rect.top))
            time.sleep(0.2)
            pyautogui.press('enter')
            app = Application(backend="win32").connect(title_re=".*PageUp/Down:前后日 空格键:操作*")

        tdx_win_tick_down.get_tick_down(stock_code)





