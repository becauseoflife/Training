from collections import deque
import cv2 as cv


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()            # getTickCount() 用于返回从操作系统启动到当前所经的计时周期数
        self._freq = 1000.0 / cv.getTickFrequency()     # getTickFrequency() 用于返回CPU的频率，这里的单位是秒，也就是一秒内重复的次数。
        self._difftimes = deque(maxlen=buffer_len)      # 使用 deque(maxlen=N) 构造函数会创建一个固定大小的队列。当新的元素加入并且这个队列已满的时候，最老的元素会自动被移除掉。

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded
