# -*- coding: utf-8 -*-


def _init():  # 初始化
    global _global_variable_dict
    _global_variable_dict = {}


def set_value(key, value):
    """ 定义一个全局变量 """
    _global_variable_dict[key] = value


def get_value(key, defaultValue=None):
    """ 获得一个全局变量,不存在则返回默认值 """
    try:
        return _global_variable_dict[key]
    except KeyError:
        return defaultValue