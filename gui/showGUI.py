import os
from tkinter import *
import tkinter

from PIL import Image, ImageTk

from args import get_global_args
from utils import getUrlFile, resize

class ShowGUI():
    def __init__(self, init_window_name):
        self.init_window_name = init_window_name
        # 获取屏幕的宽和高
        self.init_window_width = init_window_name.winfo_screenwidth()
        self.init_window_height = init_window_name.winfo_screenheight()
        self.image_index = 0
        self.image_length = 0

    def set_init_window(self):
        # 窗口名
        self.init_window_name.title("查看动作")
        # 290 160为窗口大小，+10 +10 定义窗口弹出时的默认展示位置
        size = '%dx%d+%d+%d' % (self.init_window_width - 100, self.init_window_height - 200, 0, 0)
        self.init_window_name.geometry(size)

        # 宽高
        frame_width = int(self.init_window_width / 2)
        frame_height = self.init_window_height - 100
        # 左边的框架 ===================================================================================================
        self.left_frame = Frame(self.init_window_name, bg='blue', width=1, height=frame_height)
        self.left_frame.pack(fill='both', side=tkinter.LEFT, in_=self.init_window_name)

        # 左边上部分框架 ======================================================================
        self.left_frame_top = Frame(self.left_frame, bg='blue', width=800, height=650)
        self.left_frame_top.pack(fill='both', side=tkinter.TOP, in_=self.left_frame, expand=True)
        # 图像显示提示文字
        self.picture_text_label = Label(self.left_frame_top, relief='raised', text="标准姿势识别图", width=100, height=3,
                                        font=('黑体', 12), bg='Honeydew')
        self.picture_text_label.pack(fill='both', side=tkinter.TOP, in_=self.left_frame_top)
        self.standard_picture_label = Label(self.left_frame_top, relief='groove', width=100, height=40, bg='WhiteSmoke',
                                        font=('黑体', 12))
        self.standard_picture_label.pack(fill='both', side=tkinter.BOTTOM, in_=self.left_frame_top, expand=True)

        # 按钮
        self.min_frame = Frame(self.init_window_name, bg='Snow', width=120, height=20)
        self.min_frame.pack(fill='both', side=tkinter.LEFT, in_=self.init_window_name, expand=True)
        self.up_button = Button(self.init_window_name, text='上一组', width=15, height=4, font=('黑体', 12))
        self.up_button.pack(in_=self.min_frame, expand=True)
        self.down_button = Button(self.init_window_name, text='下一组', width=15, height=4, font=('黑体', 12))
        self.down_button.pack(in_=self.min_frame, expand=True)

        # 右边的框架 ===================================================================================================
        self.right_frame = Frame(self.init_window_name, bg='blue', width=1, height=frame_height)
        self.right_frame.pack(fill='both', side=tkinter.RIGHT, in_=self.init_window_name)

        # 左边上部分框架 ======================================================================
        self.right_frame_top = Frame(self.right_frame, bg='blue', width=800, height=650)
        self.right_frame_top.pack(fill='both', side=tkinter.TOP, in_=self.right_frame, expand=True)
        # 图像显示提示文字
        self.picture_text_label = Label(self.right_frame_top, relief='raised', text="训练姿势识别图", width=100, height=3,
                                        font=('黑体', 12), bg='Honeydew')
        self.picture_text_label.pack(fill='both', side=tkinter.TOP, in_=self.right_frame_top)
        self.indentify_picture_label = Label(self.right_frame_top, relief='groove', width=100, height=40, bg='WhiteSmoke',
                                        font=('黑体', 12))
        self.indentify_picture_label.pack(fill='both', side=tkinter.BOTTOM, in_=self.right_frame_top, expand=True)

        # 绑定事件
        self.up_button.bind('<Button-1>', self.up_image)
        self.down_button.bind('<Button-1>', self.down_image)

    def up_image(self, event):
        if self.image_index - 1 >= 0:
            self.image_index = self.image_index - 1
            self.update_standard_image(self.standard_array_image[self.image_index][0])
            self.update_indentify_image(self.identify_array_image[self.image_index][0])

        # if self.image_index == 0:
        #     self.up_button.pack_forget()
        #
        # if self.image_index != 0:
        #     self.down_button.pack(in_=self.min_frame, expand=True)

    def down_image(self, event):
        if self.image_index + 1 < self.image_length:
            self.image_index = self.image_index + 1
            self.update_standard_image(self.standard_array_image[self.image_index][0])
            self.update_indentify_image(self.identify_array_image[self.image_index][0])

        # if self.image_index + 1 == self.image_length:
        #     self.down_button.pack_forget()
        #
        # if self.image_index + 1 != self.image_length:
        #     self.up_button.pack(in_=self.min_frame, expand=True)

    def init_picture(self):
        # 存储路径
        path_args = get_global_args()
        picture_store_path = path_args.picture_store_path
        identify_picture_path = path_args.identify_picture_path
        # 获取标准动作图片识别结果文件夹下的所有图片===================================
        self.standard_array_image = []
        self.identify_array_image = []
        for item in getUrlFile(picture_store_path):
            self.standard_array_image.append(item)
            # print("标准", item)
        for item in getUrlFile(identify_picture_path):
            self.identify_array_image.append(item)
            # print("识别", item)

        # 是否全部完成
        if len(self.standard_array_image) == 0 or len(self.identify_array_image) == 0 or len(self.standard_array_image) != len(self.identify_array_image):
            self.standard_picture_label.configure(text="请先完成全部康复训练!")
            self.indentify_picture_label.configure(text="请先完成全部康复训练!")
            return

        self.image_length = len(self.identify_array_image)
        # 更新图片
        self.update_standard_image(self.standard_array_image[0][0])
        self.update_indentify_image(self.identify_array_image[0][0])

        # self.up_button.pack_forget()

    # 更新图片到Label
    def update_standard_image(self, filePath):
        # 判断文件是否存在
        if os.path.exists(filePath):
            # 图片读取 #####################################################
            # image = cv.imread(filePath)
            # 转换颜色使播放时保持原有色彩
            # img = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
            # 将图像转换成Image对象
            img = Image.open(filePath)
            # current_image = Image.fromarray(img)
            self.standard_picture_label.configure(width=850, height=650)
            self.standard_picture_label.update()
            w_box = self.standard_picture_label.winfo_width()
            h_box = self.standard_picture_label.winfo_height()
            # print(w_box, "x", h_box)
            re_img = resize(880, 654, img)

            imageTK = ImageTk.PhotoImage(image=re_img)
            self.standard_picture_label.imgtk = imageTK
            self.standard_picture_label.config(image=imageTK)
            # 每执行以此只显示一张图片，需要更新窗口实现视频播放
            self.standard_picture_label.update()
        else:
            print("图片不存在")

    # 更新图片到Label
    def update_indentify_image(self, filePath):
        # 判断文件是否存在
        if os.path.exists(filePath):
            # 图片读取 #####################################################
            # image = cv.imread(filePath)
            # 转换颜色使播放时保持原有色彩
            # img = cv.cvtColor(image, cv.COLOR_BGR2RGBA)
            # 将图像转换成Image对象
            img = Image.open(filePath)
            # current_image = Image.fromarray(img)
            self.indentify_picture_label.configure(width=850, height=650)
            self.indentify_picture_label.update()
            w_box = self.indentify_picture_label.winfo_width()
            h_box = self.indentify_picture_label.winfo_height()
            # print(w_box, "x", h_box)
            re_img = resize(880, 654, img)

            imageTK = ImageTk.PhotoImage(image=re_img)
            self.indentify_picture_label.immgtk = imageTK
            self.indentify_picture_label.config(image=imageTK)
            # 每执行以此只显示一张图片，需要更新窗口实现视频播放
            self.indentify_picture_label.update()
        else:
            print("图片不存在")



# # 实例化出一个父窗口
# init_window = Tk()
# # 初始化
# ShowGUI = ShowGUI(init_window)
# # 设置根窗口默认属性
# ShowGUI.set_init_window()
# ShowGUI.init_picture()
#
# # 父窗口进入事件循环，可以理解为保持窗口运行，否则界面不展示
# init_window.mainloop()