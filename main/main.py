import argparse
from tkinter import Tk

from gui import Training_GUI
from pose import image_pose_indentify
from utils import getUrlFile

def get_args():
    parser = argparse.ArgumentParser()

    # 标准姿势图片的文件路径
    parser.add_argument('--picture_path', type=str, help='standard pose picture file path', default='../standard/image/')

    args = parser.parse_args()

    return args


def main():
    # 获取参数
    args = get_args()
    picture_path = args.picture_path

    # 实例化出一个父窗口
    init_window = Tk()
    # 初始化
    trainingGUI = Training_GUI(init_window)
    # 设置根窗口默认属性
    trainingGUI.set_init_window()

    # 获取标准动作图片文件夹下的所有图片===================================
    standard = []
    for item in getUrlFile(picture_path):
        standard.append(item)
        print(item)


    # trainingGUI.update_image("../standard/image/test2.png")
    # 进行循环处理

    image_pose_indentify(trainingGUI=trainingGUI, array_image=standard)



    # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示
    init_window.mainloop()


if __name__ == '__main__':
    main()

