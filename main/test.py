from tkinter import *
import cv2
from PIL import Image, ImageTk


def video_play():
    while video.isOpened():
        ret, frame = video.read()  # 读取照片
        # print('读取成功')
        if ret == True:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # 转换颜色使播放时保持原有色彩
            current_image = Image.fromarray(img).resize((540, 320))  # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            movieLabel.imgtk = imgtk
            movieLabel.config(image=imgtk)
            movieLabel.update()  # 每执行以此只显示一张图片，需要更新窗口实现视频播放


video = cv2.VideoCapture('../resultVideo/1.mp4')  # 使用opencv打开本地视频文件
root = Tk()
root.title('视频播放案例')
movieLabel = Label(root)  # 创建一个用于播放视频的label容器
movieLabel.pack(padx=10, pady=10)

video_play()  # 调用video_play实现视频播放

mainloop()

cv2.destroyAllWindonws()