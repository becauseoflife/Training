import os
from tkinter import *
import tkinter
from multiprocessing import Process, Queue

from PIL import Image, ImageTk

import cv2 as cv

from utils import resize


class Training_GUI():
    def __init__(self, init_window_name):
        self.init_window_name = init_window_name
        # 获取屏幕的宽和高
        self.init_window_width = init_window_name.winfo_screenwidth()
        self.init_window_height = init_window_name.winfo_screenheight()

    # 设置窗口
    def set_init_window(self):
        # 窗口名
        self.init_window_name.title("康复训练系统v1.0.0")
        # 290 160为窗口大小，+10 +10 定义窗口弹出时的默认展示位置
        # self.init_window_name.geometry('320x160+10+10')
        size = '%dx%d+%d+%d' % (self.init_window_width, self.init_window_height-100, 0, 0)
        # print(size)
        self.init_window_name.geometry(size)
        # 窗口背景色，其他背景色见：blog.csdn.net/chl0000/article/details/7657887
        # self.init_window_name["bg"] = "pink"
        # 虚化，值越小虚化程度越高
        # self.init_window_name.attributes("-alpha",0.9)
        # 三大模块

        # 菜单
        self.main_menu = Menu(self.init_window_name)
        # 菜单分组
        self.file_menu = Menu(self.main_menu)
        self.main_menu.add_cascade(label="文件", menu=self.file_menu)
        self.file_menu.add_separator()
        self.file_menu.add_cascade(label="退出", command=self.init_window_name.destroy)

        self.init_window_name.config(menu=self.main_menu)

        frame_width = int(self.init_window_width / 2)
        frame_height = self.init_window_height-100
        # 左边的框架 ===================================================================================================
        self.left_frame = Frame(self.init_window_name, bg='yellow', width=1, height=frame_height)
        self.left_frame.pack(fill='both', side=tkinter.LEFT, in_=self.init_window_name)

        # 左边上部分框架 ======================================================================
        self.left_frame_top = Frame(self.left_frame, bg='blue', width=800, height=650)
        self.left_frame_top.pack(fill='both', side=tkinter.TOP, in_=self.left_frame)
        # 图像显示提示文字
        self.picture_text_label = Label(self.left_frame_top, bg='Red', text="标准姿势", width=110, height=2)
        self.picture_text_label.pack(fill='both', side=tkinter.TOP, in_=self.left_frame_top)
        self.picture_show_label = Label(self.left_frame_top, bg='blue', width=800, height=700)
        self.picture_show_label.pack(fill='both', side=tkinter.BOTTOM, in_=self.left_frame_top)

        # 左边下部分的框架 =====================================================================
        self.left_frame_bottom = Frame(self.left_frame, bg='Red', width=110, height=32)
        self.left_frame_bottom.pack(fill='both', side=tkinter.BOTTOM, in_=self.left_frame)
        # 日志区域
        self.log_text_label = Label(self.left_frame_bottom, bg='green', text="日志输出", width=110, height=2)
        self.log_text_label.pack(fill='both', side=tkinter.TOP, in_=self.left_frame_bottom)
        self.log_output_label = Label(self.left_frame_bottom, bg='Orange', width=120, height=30)
        self.log_output_label.pack(fill='both', side=tkinter.BOTTOM, in_=self.left_frame_bottom)


        # 右边的框架 ===================================================================================================
        self.right_frame = Frame(self.init_window_name, bg='yellow', width=50, height=frame_height)
        self.right_frame.pack(fill='both', side=tkinter.RIGHT)
        # 顶部文字
        self.diplay_text_label = Label(self.right_frame, bg='blue', text="实时姿势识别", width=frame_width, height=2)
        self.diplay_text_label.pack(fill='both', side=tkinter.TOP)
        # 顶部
        self.right_frame_top = Frame(self.right_frame, bg='black', width=50, height=200)
        self.right_frame_top.pack()
        self.left_hand_show_label = Label(self.right_frame_top, bg='Red', text='左上部分身体', width=76, height=9)
        self.left_hand_show_label.pack(anchor='nw', side=tkinter.LEFT, fill='x', expand='yes')
        self.right_hand_show_label = Label(self.right_frame_top, bg='green', text='右上半部分身体', width=76, height=9)
        self.right_hand_show_label.pack(anchor='ne', side=tkinter.RIGHT, fill='x', expand='yes')
        # 中部
        self.right_frame_mid = Frame(self.right_frame, bg='orange', width=1100, height=38)
        self.right_frame_mid.pack(fill='both', after=self.right_frame_top, side=tkinter.TOP, anchor='center', expand='yes')
        self.camera_dipaly_label = Label(self.right_frame_mid, bg='purple', text="点击开始", width=1100, height=38)
        self.camera_dipaly_label.pack(fill='both', anchor='center', expand='yes')
        # 底部
        self.right_frame_end = Frame(self.right_frame, bg='green', width=50, height=200)
        self.right_frame_end.pack(fill='both', side=tkinter.BOTTOM, after=self.right_frame_mid)
        self.left_foot_show_label = Label(self.right_frame_end, bg='Red', text='左下部分身体', width=76, height=9)
        self.left_foot_show_label.pack(anchor='nw', side=tkinter.LEFT, fill='x', expand='yes')
        self.right_foot_show_label = Label(self.right_frame_end, bg='green', text='右下部分身体', width=76, height=9)
        self.right_foot_show_label.pack(anchor='ne', side=tkinter.RIGHT, fill='x', expand='yes')

        # # 实时显示区域
        # self.diplay_text_label = Label(self.right_frame, bg='blue', text="实时姿势识别", width=frame_width, height=2)
        # self.diplay_text_label.pack(fill='both', side=tkinter.TOP)
        # self.camera_dipaly_label = Label(self.right_frame_mid, bg='purple', text="点击开始", width=frame_width, height=200)
        # self.camera_dipaly_label.pack(fill='both', side=tkinter.BOTTOM)


    def update_image(self, filePath):
        # 判断文件是否存在
        if os.path.exists(filePath):
            # 图片读取 #####################################################
            # image = cv.imread(filePath)
            # 转换颜色使播放时保持原有色彩
            # img = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
            # 将图像转换成Image对象
            img = Image.open(filePath)
            # current_image = Image.fromarray(img)
            # self.picture_show_label.configure(width=850, height=650)
            self.picture_show_label.update()
            w_box = self.picture_show_label.winfo_width()
            h_box = self.picture_show_label.winfo_height()
            # print(w_box, "x", h_box)
            re_img = resize(846, 704, img)

            imageTK = ImageTk.PhotoImage(image=re_img)
            self.picture_show_label.image = imageTK
            self.picture_show_label.config(image=imageTK)
            # 每执行以此只显示一张图片，需要更新窗口实现视频播放
            self.picture_show_label.update()
        else:
            print("图片不存在")

    def update_standard_image(self, image):
        # 转换颜色使播放时保持原有色彩
        img = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
        current_image = Image.fromarray(img)
        # width, height = current_image.size
        # 大小自适应
        # w_box = self.picture_show_label.winfo_width()
        # h_box = self.picture_show_label.winfo_height()
        # print("1. ", w_box, " x ", h_box)
        re_image = resize(846, 704, current_image)

        imageTK = ImageTk.PhotoImage(image=re_image)
        self.picture_show_label.imgtk = imageTK
        self.picture_show_label.config(image=imageTK)
        # 每执行以此只显示一张图片，需要更新窗口实现视频播放
        self.picture_show_label.update()


    def update_camera_image(self, image):
        # 转换颜色使播放时保持原有色彩
        img = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
        current_image = Image.fromarray(img)
        # width, height = current_image.size
        # 大小自适应
        w_box = self.camera_dipaly_label.winfo_width()
        h_box = self.camera_dipaly_label.winfo_height()
        # print("1. ", w_box, " x ", h_box)
        re_image = resize(w_box, h_box, current_image)

        imageTK = ImageTk.PhotoImage(image=re_image)
        self.camera_dipaly_label.imgtk = imageTK
        self.camera_dipaly_label.config(image=imageTK)
        # 每执行以此只显示一张图片，需要更新窗口实现视频播放
        self.camera_dipaly_label.update()

    def update_left_hand_text(self, text):
        self.left_hand_show_label.configure(text=text)

    def update_right_hand_text(self, text):
        self.right_hand_show_label.configure(text=text)

    def update_left_foot_text(self, text):
        self.left_foot_show_label.configure(text=text)

    def update_right_foot_text(self, text):
        self.right_foot_show_label.configure(text=text)

