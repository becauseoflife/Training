# 指定编码器
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# video_writer = None
# 第一个参数是要保存的文件的路径
# fourcc 指定编码器
# fps 要保存的视频的帧率     正浮点数或正整数
# frameSize 要保存的文件的画面尺寸 (width, height)
# isColor 指示是黑白画面还是彩色的画面
# if flag is True:
#     video_writer = cv.VideoWriter(filename='../resultVideo/1.mp4', fourcc=fourcc, fps=22,
#                                   frameSize=(debug_image.shape[1], debug_image.shape[0]), isColor=True)
#     flag = False
# video_writer.write(debug_image)
# video_writer.release()

# def update_image(self, path):
#     video = cv.VideoCapture(path)
#     while True:
#         ret, image = video.read()
#         if not ret:
#             break
#         # 将图像转换成Image对象
#         img = cv.cvtColor(image, cv.COLOR_BGR2RGBA)  # 转换颜色使播放时保持原有色彩
#         current_image = Image.fromarray(img)
#         imageTK = ImageTk.PhotoImage(image=current_image)
#         self.picture_show_label.imgtk = imageTK
#         self.picture_show_label.config(image=imageTK)
#         # 每执行以此只显示一张图片，需要更新窗口实现视频播放
#         self.picture_show_label.update()
