import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont


# 位置参数说明：
# 图片
# 要添加的文字
# 文字添加到图片上的位置
# 字体的类型
# 字体大小
# 字体颜色
# 字体粗细
def cv2ImageAddText(image, text, left, top, text_color, text_size, text_bold):
    # 判断是否OpenCV图片类型
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(image)
    # 字体的格式, 宋体
    fontStyle = ImageFont.truetype("../font/simsun.ttc", text_size, encoding="utf-8")
    # text_size
    draw.text((left, top), text, text_color, fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(image), cv.COLOR_RGB2BGR)


# if __name__ == '__main__':
#     img = cv2ImageAddText(cv.imread('../font/test.jpg'), "大家好，我是片天边的云彩", 10, 65, (0, 0 , 139), 20, cv.LINE_AA)
#     cv.imshow('show', img)
#     if cv.waitKey(100000) & 0xFF == ord('q'):
#         cv.destroyAllWindows()