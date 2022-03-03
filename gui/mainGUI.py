import os
from tkinter import *
from tkinter import ttk, messagebox
import tkinter
import tkinter.messagebox
from multiprocessing import Process, Queue

from PIL import Image, ImageTk

import cv2 as cv

import globalVariable.global_variable as global_dict

from args import get_global_args
from gui.showGUI import ShowGUI
from pose import image_pose_indentify
from utils import resize, getUrlFile

class Training_GUI():
    def __init__(self, init_window_name):
        self.init_window_name = init_window_name
        # 获取屏幕的宽和高
        self.init_window_width = init_window_name.winfo_screenwidth()
        self.init_window_height = init_window_name.winfo_screenheight()

    # 设置窗口
    def set_init_window(self):
        # 窗口名
        self.init_window_name.title("运动康复训练系统 V1.0.0")
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
        self.oper_menu = Menu(self.main_menu)
        self.main_menu.add_cascade(label="功能", menu=self.oper_menu)
        self.oper_menu.add_command(label="重新开始", command=self.restart)
        self.oper_menu.add_separator()
        self.oper_menu.add_command(label="退出", command=self.window_close)

        self.show_menu = Menu(self.main_menu)
        self.main_menu.add_cascade(label="结果", menu=self.show_menu)
        self.show_menu.add_command(label="查看结果", command=self.show_result)

        self.init_window_name.config(menu=self.main_menu)

        # 宽高
        frame_width = int(self.init_window_width / 2)
        frame_height = self.init_window_height-100
        # 左边的框架 ===================================================================================================
        self.left_frame = Frame(self.init_window_name, bg='blue', width=1, height=frame_height)
        self.left_frame.pack(fill='both', side=tkinter.LEFT, in_=self.init_window_name)

        # 左边上部分框架 ======================================================================
        self.left_frame_top = Frame(self.left_frame, bg='blue', width=800, height=650)
        self.left_frame_top.pack(fill='both', side=tkinter.TOP, in_=self.left_frame, expand=True)
        # 图像显示提示文字
        self.picture_text_label = Label(self.left_frame_top, relief='solid', text="标准姿势", width=100, height=3, font=('黑体', 12, 'bold'), bg='Honeydew')
        self.picture_text_label.pack(fill='both', side=tkinter.TOP, in_=self.left_frame_top)
        self.picture_show_label = Label(self.left_frame_top, relief='solid', width=100, height=40, bg='WhiteSmoke', font=('黑体', 12, 'bold'))
        self.picture_show_label.pack(fill='both', side=tkinter.BOTTOM, in_=self.left_frame_top, expand=True)

        # 左边下部分的框架 =====================================================================
        self.left_frame_bottom = Frame(self.left_frame, bg='Red', width=110, height=32)
        self.left_frame_bottom.pack(fill='both', side=tkinter.BOTTOM, in_=self.left_frame)
        # 日志区域

        self.log_text_label = Label(self.left_frame_bottom, relief='solid', text="日志输出", width=100, height=3, font=('黑体', 12, 'bold'), bg='Honeydew')
        self.log_text_label.pack(fill='both', side=tkinter.TOP, in_=self.left_frame_bottom)
        # self.log_output_text = ttk.Treeview(self.left_frame_bottom, relief='groove', width=120, height=30, bg='WhiteSmoke', font=('宋体', 10))

        scrollbar_y = Scrollbar(self.left_frame_bottom, orient=VERTICAL)
        scrollbar_y.pack(side=tkinter.RIGHT, fill=tkinter.Y)

        style = ttk.Style()
        style.configure('mytreeview.Headings', background='gray', font=('Arial Bold', 10))

        # 日志表格
        self.log_output_table = ttk.Treeview(self.left_frame_bottom, show='headings', yscrollcommand=scrollbar_y.set)
        self.log_output_table.pack(fill='both', side=tkinter.BOTTOM, in_=self.left_frame_bottom, expand=True)

        # 定义列
        self.log_output_table["columns"] = ("时间", "标注", "序号", "腋下角度", "手肘角度", "腰部角度", "膝盖角度")
        # 设置列
        self.log_output_table.column("时间", width=120, anchor=S)
        self.log_output_table.column("标注", width=110, anchor=S)
        self.log_output_table.column("序号", width=110, anchor=S)
        self.log_output_table.column("腋下角度", width=110, anchor=S)
        self.log_output_table.column("手肘角度", width=110, anchor=S)
        self.log_output_table.column("腰部角度", width=110, anchor=S)
        self.log_output_table.column("膝盖角度", width=110, anchor=S)
        # 设置显示的表头名
        self.log_output_table.heading("时间", text="时间")
        self.log_output_table.heading("标注", text="标注")
        self.log_output_table.heading("序号", text="序号")
        self.log_output_table.heading("腋下角度", text="左/右腋下角度")
        self.log_output_table.heading("手肘角度", text="左/右手肘角度")
        self.log_output_table.heading("腰部角度", text="左/右腰部角度")
        self.log_output_table.heading("膝盖角度", text="左/右膝盖角度")
        # 滚动条
        # self.log_output_table.configure(yscrollcommand=scrollbar_y)
        scrollbar_y.config(command=self.log_output_table.yview)


        # 右边的框架 ===================================================================================================
        self.right_frame = Frame(self.init_window_name, bg='yellow', width=50, height=frame_height)
        self.right_frame.pack(fill='both', side=tkinter.RIGHT, in_=self.init_window_name)
        # 顶部文字
        self.diplay_text_label = Label(self.right_frame, relief='solid', text="实时姿势识别", width=frame_width, height=3, font=('黑体', 12, 'bold'), bg='Honeydew')
        self.diplay_text_label.pack(fill='both', side=tkinter.TOP, in_=self.right_frame)
        # 顶部
        self.right_frame_top = Frame(self.right_frame, bg='black', width=50, height=200)
        self.right_frame_top.pack(in_=self.right_frame)
        self.left_hand_show_label = Label(self.right_frame_top, relief='solid', text='左上半部分身体提示区', width=57, height=9, font=('黑体', 12, 'bold'), bg='Snow', fg='red')
        self.left_hand_show_label.pack(anchor='nw', side=tkinter.LEFT, fill='x', expand='yes', in_=self.right_frame_top)
        self.right_hand_show_label = Label(self.right_frame_top, relief='solid', text='右上半部分身体提示区', width=60, height=9, font=('黑体', 12, 'bold'), bg='Snow', fg='red')
        self.right_hand_show_label.pack(anchor='ne', side=tkinter.RIGHT, fill='x', expand='yes', in_=self.right_frame_top)
        # 中部
        self.right_frame_mid = Frame(self.right_frame, bg='orange', width=1100, height=38)
        self.right_frame_mid.pack(fill='both', after=self.right_frame_top, side=tkinter.TOP, anchor='center', expand='yes', in_=self.right_frame)
        self.camera_dipaly_label = Label(self.right_frame_mid, relief='solid', text="点击开始", width=1100, height=38, font=('黑体', 12, 'bold'), bg='WhiteSmoke')
        self.camera_dipaly_label.pack(fill='both', anchor='center', expand='yes', in_=self.right_frame_mid)
        # 底部
        self.right_frame_end = Frame(self.right_frame, relief='groove', width=50, height=200)
        self.right_frame_end.pack(fill='both', side=tkinter.BOTTOM, after=self.right_frame_mid, in_=self.right_frame)
        self.left_foot_show_label = Label(self.right_frame_end, relief='solid', text='左下部分身体提示区', width=57, height=9, font=('黑体', 12, 'bold'), bg='Snow', fg='red')
        self.left_foot_show_label.pack(anchor='nw', side=tkinter.LEFT, fill='x', expand='yes', in_=self.right_frame_end)
        self.right_foot_show_label = Label(self.right_frame_end, relief='solid', text='右下部分身体提示区', width=60, height=9, font=('黑体', 12, 'bold'), bg='Snow', fg='red')
        self.right_foot_show_label.pack(anchor='ne', side=tkinter.RIGHT, fill='x', expand='yes', in_=self.right_frame_end)

        # 绑定事件
        self.camera_dipaly_label.bind('<Button-1>', self.left_mouse_down)
        self.init_window_name.protocol("WM_DELETE_WINDOW", self.window_close)

    def left_mouse_down(self, event):
        # print('鼠标左键按下')
        self.camera_dipaly_label.configure(text="加载中，请稍等···")
        # self.picture_show_label.configure(text="等待加载摄像设备···")
        self.camera_dipaly_label.update()
        self.camera_dipaly_label.unbind('<Button-1>')
        image_pose_indentify()

    # 重新开始
    def restart(self):
        self.log_output_table.insert('', END, values=['', '', '', '', '', ''])
        self.camera_dipaly_label.configure(text="加载中，请稍等···")
        # self.picture_show_label.configure(text="等待加载摄像设备···")
        self.camera_dipaly_label.update()
        self.camera_dipaly_label.unbind('<Button-1>')
        image_pose_indentify()

    # 展示结果
    def show_result(self):
        # 实例化出一个父窗口
        init_window = Toplevel()
        # 初始化
        showGUI = ShowGUI(init_window)
        # 设置根窗口默认属性
        showGUI.set_init_window()
        showGUI.init_picture()
        # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示
        init_window.mainloop()


    def window_close(self):
        if messagebox.askokcancel("退出", "确认退出?"):
            cap = global_dict.get_value('cap')
            if cap is not None:
                cap.release()
            # 销毁界面
            self.init_window_name.destroy()

    # 识别区域的文字
    def set_text(self, text):
        self.camera_dipaly_label.configure(text=text)
        self.picture_show_label.configure(text=text)

    # 更新图片到Label
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
            self.picture_show_label.configure(width=850, height=650)
            self.picture_show_label.update()
            w_box = self.picture_show_label.winfo_width()
            h_box = self.picture_show_label.winfo_height()
            print(w_box, "x", h_box)
            re_img = resize(880, 654, img)

            imageTK = ImageTk.PhotoImage(image=re_img)
            self.picture_show_label.image = imageTK
            self.picture_show_label.config(image=imageTK)
            # 每执行以此只显示一张图片，需要更新窗口实现视频播放
            self.picture_show_label.update()
        else:
            print("图片不存在")

    # 标准图片处理后更新到界面上
    def update_standard_image(self, image):
        # 转换颜色使播放时保持原有色彩
        img = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
        current_image = Image.fromarray(img)
        # width, height = current_image.size
        # 大小自适应
        # w_box = self.picture_show_label.winfo_width()
        # h_box = self.picture_show_label.winfo_height()
        # print("1. ", w_box, " x ", h_box)
        re_image = resize(880, 654, current_image)

        imageTK = ImageTk.PhotoImage(image=re_image)
        self.picture_show_label.imgtk = imageTK
        self.picture_show_label.config(image=imageTK)
        # 每执行以此只显示一张图片，需要更新窗口实现视频播放
        self.picture_show_label.update()

    # 相机获取的图片帧更新到组件上
    def update_camera_image(self, image):
        # 转换颜色使播放时保持原有色彩
        img = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
        current_image = Image.fromarray(img)
        # width, height = current_image.size
        # 大小自适应
        w_box = self.camera_dipaly_label.winfo_width()
        h_box = self.camera_dipaly_label.winfo_height()
        re_image = resize(w_box, h_box, current_image)

        imageTK = ImageTk.PhotoImage(image=re_image)
        self.camera_dipaly_label.imgtk = imageTK
        self.camera_dipaly_label.config(image=imageTK)
        # 每执行以此只显示一张图片，需要更新窗口实现视频播放
        self.camera_dipaly_label.update()

    # 更新左上部分身体提示文字
    def update_left_hand_text(self, text):
        self.left_hand_show_label.configure(text=text)

    # 更新右上部分身体提示文字
    def update_right_hand_text(self, text):
        self.right_hand_show_label.configure(text=text)

    # 更新左下部分身体提示文字
    def update_left_foot_text(self, text):
        self.left_foot_show_label.configure(text=text)

    # 更新右下部分身体提示文字
    def update_right_foot_text(self, text):
        self.right_foot_show_label.configure(text=text)

    # 展示完成提示框
    def show_info(self):
        tkinter.messagebox.showinfo('提示', '                    训练完成！                    \n')

    # 添加日志
    # 添加数据到末尾
    def add_log(self, data):
        self.log_output_table.insert('', END, values=data)

    # 添加空行数据日志
    def add_log_empty(self, data):
        self.log_output_table.insert('', END, values=data)
        self.log_output_table.insert('', END, values=['', '', '', '', '', ''])

    # 识别完成
    def identify_compass(self):
        self.picture_show_label.config(image='')
        self.picture_show_label.update()
        self.camera_dipaly_label.config(image='')
        self.camera_dipaly_label.update()
        self.camera_dipaly_label.configure(text="点击开始")
        self.left_hand_show_label.configure(text="左上半部分身体提示区")
        self.right_hand_show_label.configure(text='右上半部分身体提示区')
        self.left_foot_show_label.configure(text='左下半部分身体提示区')
        self.right_foot_show_label.configure(text='右下半部分身体提示区')

        # 绑定事件
        self.camera_dipaly_label.bind('<Button-1>', self.left_mouse_down)



