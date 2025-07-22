import pyautogui
from pywinauto import Application
import time
import re

from datetime import datetime

def extract_date(text, return_datetime=False):
    """
    从字符串中提取日期信息（格式：XXXX年XX月XX日）

    参数：
        text (str): 待提取的字符串
        return_datetime (bool): 是否返回 datetime 对象，默认为 False（返回字符串）

    返回：
        str/datetime: 提取的日期字符串或 datetime 对象（若未找到返回 None）
    """
    # 匹配 "XXXX年XX月XX日" 格式的日期
    date_pattern = r"\d{4}年\d{2}月\d{2}日"
    match = re.search(date_pattern, text)

    if not match:
        return None

    date_str = match.group()

    if return_datetime:
        try:
            return datetime.strptime(date_str, "%Y年%m月%d日").date()
        except ValueError:
            return None
    else:
        return date_str

def extract_code(text):
    match = re.search(r"\((.*?)\)", text)  # 匹配括号内的内容

    if match:
        return match.group(1)  # group(1) 提取括号内的部分
    else:
        return None