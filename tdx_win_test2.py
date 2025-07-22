import pyautogui
import win32gui
from pywinauto import Application
import time
from tdx_win_unit import extract_date


from pywinauto import Desktop

# # 获取所有顶层窗口
# windows = Desktop(backend="win32").windows()
#
# for w in windows:
#     print(f"标题: {w.window_text()}, 类名: {w.class_name()}, 句柄: {w.handle}")
#
# # 过滤可见窗口
# visible_windows = [w for w in windows if w.is_visible()]
app = Application(backend="win32").connect(title_re=".*通达信专业研究版V7.*")
tdx_window = app.window(class_name="TdxW_MainFrame_Class")
child = tdx_window.print_control_identifiers()  # 查找菜单相关控件
print(child)


# app = Application(backend="win32").connect(title_re=".*PageUp/Down:前后日 空格键:操作*")  # 匹配含"通达信"的窗口
# tdx_window = app.window(title_re=".*PageUp/Down:前后日 空格键:操作*")
# print(tdx_window.window_text())
# print(extract_date(tdx_window.window_text(),return_datetime = True))

# def find_tdx_window():
#     def callback(hwnd, extra):
#         if "通达信" in win32gui.GetWindowText(hwnd):
#             print(f"找到窗口：{win32gui.GetWindowText(hwnd)}, 句柄：{hwnd}")
#             extra.append(hwnd)
#     hwnds = []
#     win32gui.EnumWindows(callback, hwnds)
#     return hwnds[0] if hwnds else None
#
# tdx_hwnd = find_tdx_window()
# if tdx_hwnd:
#     print("找到通达信窗口，句柄：", tdx_hwnd)
# else:
#     print("未找到通达信窗口！")

# child = tdx_window.print_control_identifiers()  # 查找菜单相关控件
# print(child)
# print('==========================')


# # 打印所有顶层窗口的类名
# for w in app.windows():
#     # print(f"窗口标题: {w.window_text()}, 类名: {w.class_name()},类名2:{w.class_name}")
#     print(f"窗口标题: {w.window_text()}, 类名: {w.class_name()}")
#
#
# popup_menu = app.window(title="TdxW",class_name="#32770")
# if popup_menu.exists():
#     print("存在")
# else:
#     print("不存在")