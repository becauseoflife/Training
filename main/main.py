import argparse
from tkinter import Tk

from args import get_global_args
from gui import Training_GUI
from pose import image_pose_indentify
from utils import getUrlFile

import globalVariable.global_variable as global_dict


def main():
    # 初始化全局变量
    global_dict._init()

    # 实例化出一个父窗口
    init_window = Tk()
    # 初始化
    trainingGUI = Training_GUI(init_window)
    # 设置根窗口默认属性
    trainingGUI.set_init_window()

    # 设置界面为全局变量
    global_dict.set_value('trainingGUI', trainingGUI)

    # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示
    init_window.mainloop()


if __name__ == '__main__':
    main()

