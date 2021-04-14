import os


def getUrlFile(path):
    # 指定路径
    path = path
    # 返回指定路径的文件夹名称
    dirs = os.listdir(path)
    # 循环遍历该目录下的照片
    for dir in dirs:
        # 拼接字符串
        pa = path + dir
        # 判断是否为照片
        if not os.path.isdir(pa):
            # 使用生成器循环输出
            yield pa, dir


if __name__ == '__main__':
    for item in getUrlFile(r'../standard/image/'):
        print(item)