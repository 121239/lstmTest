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


    for i in range(300):
        print(extract_date(tdx_window.window_text(), return_datetime=True))

        # 点击控件中心
        tdx_window.right_click_input(coords=(center_x - rect.left, center_y - rect.top))  # 相对坐标
        time.sleep(0.5)
        if(i == 0) :
            pyautogui.press('pageup')
            time.sleep(2)

        # todo 第一次的时候执行 循环向上移动直到时间是最新的时间 或者是本周五 能找到上一个工作日吗 可以试试
        #  这个现在也不重要 先弄AI AIAIAIAIAI!!!!!!!!!

        # 捕获弹出菜单（根据实际类名调整）
        popup_menu = app.window(class_name="#32768")
        if popup_menu.exists():
            # 键盘操作 导出分笔数据
            pyautogui.press('down', presses=13)  # 按↓键13次
            pyautogui.press('enter')  # 回车确认
            time.sleep(0.5)
            pyautogui.press('enter')
            time.sleep(0.5)
            if app.window(title="TdxW", class_name="#32770").exists():
                pyautogui.press('esc')  # 通用取消 取消弹框
            else:
                print("数据导出失败")
        else:
            print("菜单未弹出！")
            # continue

        tdx_window.click_input(coords=(center_x - rect.left, center_y - rect.top))
        pyautogui.press('pageup')
        time.sleep(2)

    # 取消多余窗口
    pyautogui.press('esc')
    pyautogui.press('esc')
    pyautogui.press('esc')
    pyautogui.press('esc')