#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import math

import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image

from utils import CvFpsCalc, LoadData
from utils import interval
from utils import cv2ImageAddText

from pose.image_pose_data import get_picture_angle

from multiprocessing import Process, Queue


# argparse 模块可以让人轻松编写用户友好的命令行接口。
# 程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
# argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
# 此模块是 Python 标准库中推荐的命令行解析模块。
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=3000)
    parser.add_argument("--height", help='cap height', type=int, default=2000)
    # 如果设置为true，则解决方案仅输出25个上身姿势界标。否则，它将输出33个姿势地标的完整集合。
    # 请注意，对于大多数下半身看不见的用例，仅上半身的预测可能会更准确。预设为false。
    parser.add_argument('--upper_body_only', action='store_true')
    # [0.0, 1.0]来自人员检测模型的最小置信度值（）被认为是成功的检测。预设为0.5。
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    # [0.0, 1.0]来自地标跟踪模型的姿势地标的最小置信度值（）将被视为已成功跟踪，否则将在下一个输入图像上自动调用人检测。
    # 将其设置为更高的值可以提高解决方案的健壮性，但代价是更高的延迟。
    # 如果static_image_mode是true，则忽略该位置，其中人检测仅在每个图像上运行。预设为0.5。
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args


# 传入标准的姿势图片数组
def image_pose_indentify(trainingGUI, array_image):
    # 参数分析v #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    upper_body_only = args.upper_body_only
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # 照相机准备 ###############################################################
    #  VideoCapture()中参数是0，表示打开笔记本的内置摄像头；
    # 参数是1，则打开外置摄像头；
    # 其他数字则代表其他设备；
    # 参数是视频文件的路径则打开指定路径下的视频。
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 模型加载 #############################################################
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        upper_body_only=upper_body_only,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS测量模块 ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 读取出来的图像数据
    # json_util = LoadData('angle')
    # json_data = json_util.load_data()
    # print(json_data)
    # data_arr = json_data['data']
    # print(data_arr)
    # # print(data_arr[0])
    # # print(data_arr[1])
    # # print(data_arr[0]['armpit'])
    # print(len(data_arr))

    # 指定编码器
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # video_writer = None
    # flag = True

    print("要处理的突变数组长度: ", len(array_image))
    print("参数: ", array_image)
    # 依次执行每张图片
    index = 0
    update_flag = 0
    # 图片个数
    imgArrLen = len(array_image)

    while index < imgArrLen:
        display_fps = cvFpsCalc.get()

        # 相机抓取 #####################################################
        # 参数ret 为 True 或者 False,代表有没有读取到图片
        # 第二个参数 image 表示截取到一帧的图片
        ret, image = cap.read()
        if not ret:
            break
        # 镜像显示
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # 检测实施 #############################################################
        # 由于OpenCV默认的颜色空间是BGR，但是一般我们说的颜色空间为RGB，这里mediapipe便修改了颜色空间
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        # 获取图片标准姿势的数据
        if update_flag == index:
            update_flag = update_flag + 1
            # print(array_image[index][0])
            data_arr, standard_image = get_picture_angle(array_image[index][0])

            # 显示标准姿势识别的图片
            trainingGUI.update_standard_image(standard_image)

        # 要显示的提示文字包括 左、右上半部分身体；左、右下半部分身体
        result_up_l = ""
        result_up_r = ""
        result_down_l = ""
        result_down_r = ""

        # 绘画 ################################################################
        if results.pose_landmarks is not None:
            # 外接矩形的计算
            brect = calc_bounding_rect(debug_image, results.pose_landmarks)
            # 绘画
            debug_image, landmark_points = draw_landmarks(debug_image, results.pose_landmarks)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            # 获取身体角度数据
            angle_json = calc_angle(landmark_points)
            # print(angle_json)

            # 识别
            # print(data_arr['data'])
            tipIncText_l, tipIncText_r, tipDecText_l, tipDecText_r = compareAngle(data_arr['data'], angle_json, 5)

            if len(tipIncText_l) == 0 and len(tipIncText_r) == 0 and len(tipDecText_l) == 0 and len(tipDecText_r) == 0:
                print("识别的角度： ", angle_json)
                print("标准的角度： ", data_arr['data'])
                print("识别成功！==============", index, "===============")
                index = index + 1

            # 处理显示的文字
            for s in tipIncText_l:
                result_up_l += s + "\n"
            for s in tipIncText_r:
                result_up_r += s + "\n"
            for s in tipDecText_l:
                result_down_l += s + "\n"
            for s in tipDecText_r:
                result_down_r += s + "\n"
            # 显示到界面上
            trainingGUI.update_left_hand_text(result_up_l)
            trainingGUI.update_right_hand_text(result_up_r)
            trainingGUI.update_left_foot_text(result_down_l)
            trainingGUI.update_right_foot_text(result_down_r)

            cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

            # 显示相机识别的图片到界面上
            trainingGUI.update_camera_image(debug_image)

        # 位置参数说明：
        # 图片
        # 要添加的文字
        # 文字添加到图片上的位置
        # 字体的类型
        # 字体大小
        # 字体颜色
        # 字体粗细
        # debug_image = cv2ImageAddText(debug_image, result_str_l, 20, 40, (0, 255, 255), 20, cv.LINE_AA)
        # debug_image = cv2ImageAddText(debug_image, result_str_r, 1000, 40, (0, 255, 255), 20, cv.LINE_AA)

        # 第一个参数是要保存的文件的路径
        # fourcc 指定编码器
        # fps 要保存的视频的帧率     正浮点数或正整数
        # frameSize 要保存的文件的画面尺寸 (width, height)
        # isColor 指示是黑白画面还是彩色的画面
        # if flag is True:
        #     video_writer = cv.VideoWriter(filename='../resultVideo/test.avi', fourcc=fourcc, fps=5,
        #                                   frameSize=(debug_image.shape[1], debug_image.shape[0]), isColor=True)
        #     flag = False
        # video_writer.write(debug_image)

        # 按键处理（ESC：结束） #################################################
        # key = cv.waitKey(1)
        # if key == 27:  # ESC
        #     break

        # 屏幕反射 #############################################################
        # cv.imshow('MediaPipe Pose Demo', debug_image)

    # video_writer.release()
    cap.release()
    cv.destroyAllWindows()

    # return queue


# 比较两个姿势角度数据是否相等
def compareAngle(standard_angle, angle_json, mistake=10):
    # 判断是否识别
    flag = True
    # 提示文字
    tipIncText_l = []
    tipIncText_r = []

    tipDecText_l = []
    tipDecText_r = []

    # 手臂和腰
    standard = standard_angle['armpit']['left_armpit_angle']
    angle = angle_json['armpit']['left_armpit_angle']
    if angle != -1:
        if standard != -1 and angle not in interval(standard, mistake):
            if angle < standard - mistake:
                tipIncText_l.append("请增大左手臂与左侧腰之间的幅度")
            if angle > standard + mistake:
                tipIncText_l.append("请减小左手臂与左侧腰之间的幅度")
    elif angle == -1 and standard != -1:
        tipIncText_l.append("未识别到左手臂和腰")


    standard = standard_angle['armpit']['right_armpit_angle']
    angle = angle_json['armpit']['right_armpit_angle']
    if angle != -1:
        if standard != -1 and angle not in interval(standard, mistake):
            if angle < standard - mistake:
                tipIncText_r.append("请增大右手臂与右侧腰之间的幅度！")
            if angle > standard + mistake:
                tipIncText_r.append("请减小右手臂与右侧腰之间的幅度！")
    elif angle == -1 and standard != -1:
        tipIncText_r.append("未识别到右手臂和腰")

    # 肩膀的手臂
    # standard = standard_angle['shoulder']['left_shoulder_angle']
    # angle = angle_json['shoulder']['left_shoulder_angle']
    # if standard != -1 and angle not in interval(standard, mistake):
    #     if angle < standard - mistake:
    #         tipIncText.append("请增大肩膀与左臂之间的幅度！")
    #     if angle > standard + mistake:
    #         tipDecText.append("请减小肩膀与左臂之间的幅度！")
    #
    # standard = standard_angle['shoulder']['right_shoulder_angle']
    # angle = angle_json['shoulder']['right_shoulder_angle']
    # if standard != -1 and angle not in interval(standard, mistake):
    #     if angle < standard - mistake:
    #         tipIncText.append("请增大肩膀与右臂之间的幅度！")
    #     if angle < standard + mistake:
    #         tipDecText.append("请减小肩膀与右臂之间的幅度！")

    # 手肘
    standard = standard_angle['elbow']['left_elbow_angle']
    angle = angle_json['elbow']['left_elbow_angle']
    if angle != -1:
        if standard != -1 and angle not in interval(standard, mistake):
            if angle < standard - mistake:
                tipIncText_l.append("请增大左手臂的幅度！")
            if angle > standard + mistake:
                tipIncText_l.append("请减小左手臂的幅度！")
    elif angle == -1 and standard != -1:
        tipIncText_l.append("未识别到左手肘")

    standard = standard_angle['elbow']['right_elbow_angle']
    angle = angle_json['elbow']['right_elbow_angle']
    if angle != -1:
        if standard != -1 and angle not in interval(standard, mistake):
            if angle < standard - mistake:
                tipIncText_r.append("请增大右手臂的幅度！")
            if angle > standard + mistake:
                tipIncText_r.append("请减小右手臂的幅度！")
    elif angle == -1 and standard != -1:
        tipIncText_r.append("未识别到右手肘")


    # 腰部
    standard = standard_angle['hip']['left_hip_angle']
    angle = angle_json['hip']['left_hip_angle']
    if angle != -1:
        if standard != -1 and angle not in interval(standard, mistake):
            if angle < standard - mistake:
                tipDecText_l.append("请增大左侧腰部和左大腿之间的幅度！")
            if angle > standard + mistake:
                tipDecText_l.append("请减小左侧腰部和左大腿之间的幅度！")
    elif angle == -1 and standard != -1:
        tipDecText_l.append("未识别左腰部")

    standard = standard_angle['hip']['right_hip_angle']
    angle = angle_json['hip']['right_hip_angle']
    if angle != -1:
        if standard != -1 and angle not in interval(standard, mistake):
            if angle < standard - mistake:
                tipDecText_r.append("请增大右侧腰部和右大腿之间的幅度！")
            if angle > standard + mistake:
                tipDecText_r.append("请减小右侧腰部和右大腿之间的幅度！")
    elif angle == -1 and standard != -1:
        tipDecText_r.append("未识别右腰部")

    # 膝盖
    standard = standard_angle['knee']['left_knee_angle']
    angle = angle_json['knee']['left_knee_angle']
    if angle != -1:
        if standard != -1 and angle not in interval(standard, mistake):
            if angle < standard - mistake:
                tipDecText_l.append("请增大左膝盖的幅度！")
            if angle > standard + mistake:
                tipDecText_l.append("请减小左膝盖的幅度！")
    elif angle == -1 and standard != -1:
        tipDecText_l.append("未识别左膝盖")

    standard = standard_angle['knee']['right_knee_angle']
    angle = angle_json['knee']['right_knee_angle']
    if angle != -1:
        if standard != -1 and angle not in interval(standard, mistake):
            if angle < standard - mistake:
                tipDecText_r.append("请增大右膝盖的幅度！")
            if angle > standard + mistake:
                tipDecText_r.append("请减小右膝盖的幅度！")
    elif angle == -1 and standard != -1:
        tipDecText_r.append("未识别右膝盖")

    return tipIncText_l, tipIncText_r, tipDecText_l, tipDecText_r


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    # 这个函数可以创建一个没有任何具体值的 array 数组，是创建数组最快的方法。
    # 根据给定的维度和数值类型返回一个新的数组，其元素不进行初始化。
    landmark_array = np.empty((0, 2), int)

    # enumerate(sequence, [start=0])
    # sequence -- 一个序列、迭代器或其他支持迭代对象。
    # start -- 下标起始位置。
    # _用作被丢弃的名称。
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    # 矩形边框（Bounding Rectangle）是说，用一个最小的矩形，把找到的形状包起来。
    # 返回四个值，分别是x，y，w，h；
    # x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, landmarks, visibility_th=0.5):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        # · visibility：一个值，用于[0.0, 1.0]指示界标在图像中可见（存在且未被遮挡）的可能性。
        if landmark.visibility < visibility_th:
            continue

        # cvCircle(CvArr* img, CvPoint center, int radius, CvScalar color, int thickness=1, int lineType=8, int shift=0)
        # img为源图像指针
        # center为画圆的圆心坐标
        # radius为圆的半径
        # color为设定圆的颜色，规则根据B（蓝）G（绿）R（红）
        # thickness 如果是正数，表示组成圆的线条的粗细程度。否则，表示圆是否被填充#
        # line_type 线条的类型。默认是8
        # shift 圆心坐标点和半径值的小数点位数

        if index == 0:  # 鼻
            cv.circle(image, (landmark_x, landmark_y), 5, (255, 255, 0), 2)
        if index == 1:  # 右眼：眼内角
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 右眼：瞳孔
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 右眼：眼睛的外角
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 左眼：眼内角
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 左眼：瞳孔
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 左眼：眼睛的外角
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 右耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 左耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 右肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  # 右肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 右手腕
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 左手腕
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  # 右手1（外缘）
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 左手1（外缘）
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 右手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 左手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 右手3（内边缘）
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 左手3（内边缘）
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 腰(右側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:  # 腰(左側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  # 右膝
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  # 左膝
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:  # 右脚踝
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  # 左脚踝
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  # 右脚跟
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 左脚跟
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 右脚趾
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 左脚趾
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

    # cv.line（） 函数
    # 第一个参数 img：要划的线所在的图像;
    # 　　第二个参数 pt1：直线起点
    # 　　第三个参数 pt2：直线终点
    # 　　第四个参数 color：直线的颜色
    # 　　第五个参数 thickness=1：线条粗细

    if len(landmark_point) > 0:
        # 右眼
        if landmark_point[1][0] > visibility_th and landmark_point[2][
            0] > visibility_th:
            cv.line(image, landmark_point[1][1], landmark_point[2][1],
                    (0, 255, 0), 2)
        if landmark_point[2][0] > visibility_th and landmark_point[3][
            0] > visibility_th:
            cv.line(image, landmark_point[2][1], landmark_point[3][1],
                    (0, 255, 0), 2)

        # 左眼
        if landmark_point[4][0] > visibility_th and landmark_point[5][
            0] > visibility_th:
            cv.line(image, landmark_point[4][1], landmark_point[5][1],
                    (0, 255, 0), 2)
        if landmark_point[5][0] > visibility_th and landmark_point[6][
            0] > visibility_th:
            cv.line(image, landmark_point[5][1], landmark_point[6][1],
                    (0, 255, 0), 2)

        # 口
        if landmark_point[9][0] > visibility_th and landmark_point[10][
            0] > visibility_th:
            cv.line(image, landmark_point[9][1], landmark_point[10][1],
                    (0, 255, 0), 2)

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][
            0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[12][1],
                    (255, 255, 0), 2)

        # 右臂
        if landmark_point[11][0] > visibility_th and landmark_point[13][
            0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[13][1],
                    (0, 255, 0), 2)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
            cv.line(image, landmark_point[13][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左臂
        if landmark_point[12][0] > visibility_th and landmark_point[14][
            0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1],
                    (0, 255, 255), 2)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1],
                    (0, 255, 255), 2)

        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][
            0] > visibility_th:
            cv.line(image, landmark_point[15][1], landmark_point[17][1],
                    (0, 255, 0), 2)
        if landmark_point[17][0] > visibility_th and landmark_point[19][
            0] > visibility_th:
            cv.line(image, landmark_point[17][1], landmark_point[19][1],
                    (0, 255, 0), 2)
        if landmark_point[19][0] > visibility_th and landmark_point[21][
            0] > visibility_th:
            cv.line(image, landmark_point[19][1], landmark_point[21][1],
                    (0, 255, 0), 2)
        if landmark_point[21][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
            cv.line(image, landmark_point[21][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][
            0] > visibility_th:
            cv.line(image, landmark_point[16][1], landmark_point[18][1],
                    (0, 255, 0), 2)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
            0] > visibility_th:
            cv.line(image, landmark_point[18][1], landmark_point[20][1],
                    (0, 255, 0), 2)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
            0] > visibility_th:
            cv.line(image, landmark_point[20][1], landmark_point[22][1],
                    (0, 255, 0), 2)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
            cv.line(image, landmark_point[22][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 身体
        if landmark_point[11][0] > visibility_th and landmark_point[23][
            0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[23][1],
                    (0, 255, 0), 2)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[24][1],
                    (0, 255, 0), 2)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[24][1],
                    (0, 255, 0), 2)

        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[25][1],
                        (0, 255, 0), 2)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                0] > visibility_th:
                cv.line(image, landmark_point[25][1], landmark_point[27][1],
                        (0, 255, 0), 2)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                0] > visibility_th:
                cv.line(image, landmark_point[27][1], landmark_point[29][1],
                        (0, 255, 0), 2)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                0] > visibility_th:
                cv.line(image, landmark_point[29][1], landmark_point[31][1],
                        (0, 255, 0), 2)

            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                0] > visibility_th:
                cv.line(image, landmark_point[24][1], landmark_point[26][1],
                        (0, 255, 0), 2)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                0] > visibility_th:
                cv.line(image, landmark_point[26][1], landmark_point[28][1],
                        (0, 255, 0), 2)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                0] > visibility_th:
                cv.line(image, landmark_point[28][1], landmark_point[30][1],
                        (0, 255, 0), 2)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                0] > visibility_th:
                cv.line(image, landmark_point[30][1], landmark_point[32][1],
                        (0, 255, 0), 2)
    return image, landmark_point


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (255, 255, 0), 2)

    return image


def calc_angle(landmark_points, visibility_th=0.5):
    # pose landmarks ： https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png

    body_angles_json = {}

    # 手臂和身体的夹角，公共点 肩部
    left_armpit_angle = -1
    right_armpit_angle = -1
    if landmark_points[13][0] > visibility_th and landmark_points[23][0] and landmark_points[11][0] > visibility_th:
        left_armpit_angle = angle(landmark_points[13][1], landmark_points[23][1], landmark_points[11][1])
    if landmark_points[14][0] > visibility_th and landmark_points[24][0] > visibility_th and landmark_points[12][
        0] > visibility_th:
        right_armpit_angle = angle(landmark_points[14][1], landmark_points[24][1], landmark_points[12][1])
    # body_angles.append(['armpit', (left_armpit_angle, right_armpit_angle)])
    body_angles_json['armpit'] = {'left_armpit_angle': left_armpit_angle, 'right_armpit_angle': right_armpit_angle}

    # 手臂和肩膀的夹角，公共点 肩部
    left_shoulder_angle = -1
    right_shoulder_angle = -1
    if landmark_points[12][0] > visibility_th and landmark_points[13][0] > visibility_th and landmark_points[11][
        0] > visibility_th:
        left_shoulder_angle = angle(landmark_points[12][1], landmark_points[13][1], landmark_points[11][1])
    if landmark_points[11][0] > visibility_th and landmark_points[14][0] > visibility_th and landmark_points[12][
        0] > visibility_th:
        right_shoulder_angle = angle(landmark_points[11][1], landmark_points[14][1], landmark_points[12][1])
    # body_angles.append(['shoulder', (left_shoulder_angle, right_shoulder_angle)])
    body_angles_json['shoulder'] = {'left_shoulder_angle': left_shoulder_angle,
                                    'right_shoulder_angle': right_shoulder_angle}

    # 肩膀和手腕的夹角，公共点 肘部
    left_elbow_angle = -1
    right_elbow_angle = -1
    if landmark_points[11][0] > visibility_th and landmark_points[15][0] > visibility_th and landmark_points[13][
        0] > visibility_th:
        left_elbow_angle = angle(landmark_points[11][1], landmark_points[15][1], landmark_points[13][1])
    if landmark_points[12][0] > visibility_th and landmark_points[16][0] > visibility_th and landmark_points[14][
        0] > visibility_th:
        right_elbow_angle = angle(landmark_points[12][1], landmark_points[16][1], landmark_points[14][1])
    # body_angles.append(['elbow', (left_elbow_angle, right_elbow_angle)])
    body_angles_json['elbow'] = {'left_elbow_angle': left_elbow_angle, 'right_elbow_angle': right_elbow_angle}

    # 身体和大腿的夹角，公共点 臀部
    left_hip_angle = -1
    right_hip_angle = -1
    if landmark_points[11][0] > visibility_th and landmark_points[25][0] > visibility_th and landmark_points[23][
        0] > visibility_th:
        left_hip_angle = angle(landmark_points[11][1], landmark_points[25][1], landmark_points[23][1])
    if landmark_points[12][0] > visibility_th and landmark_points[26][0] > visibility_th and landmark_points[24][
        0] > visibility_th:
        right_hip_angle = angle(landmark_points[12][1], landmark_points[26][1], landmark_points[24][1])
    # body_angles.append(['hip', (left_hip_angle, right_hip_angle)])
    body_angles_json['hip'] = {'left_hip_angle': left_hip_angle, 'right_hip_angle': right_hip_angle}

    # 大腿和小腿之间的夹角，公共点 膝盖
    left_knee_angle = -1
    right_knee_angle = -1
    if landmark_points[23][0] > visibility_th and landmark_points[27][0] > visibility_th and landmark_points[25][
        0] > visibility_th:
        left_knee_angle = angle(landmark_points[23][1], landmark_points[27][1], landmark_points[25][1])
    if landmark_points[24][0] > visibility_th and landmark_points[28][0] > visibility_th and landmark_points[26][
        0] > visibility_th:
        right_knee_angle = angle(landmark_points[24][1], landmark_points[28][1], landmark_points[26][1])
    # body_angles.append(['knee', (left_knee_angle, right_knee_angle)])
    body_angles_json['knee'] = {'left_knee_angle': left_knee_angle, 'right_knee_angle': right_knee_angle}

    return body_angles_json


def angle(point1, point2, public_point):
    x1 = point1[0] - public_point[0]
    y1 = point1[1] - public_point[1]
    x2 = point2[0] - public_point[0]
    y2 = point2[1] - public_point[1]

    # atan2 返回给定的 X 及 Y 坐标值的反正切值。
    angle1 = math.atan2(y1, x1)
    angle1 = int(angle1 * 180 / math.pi)
    # print("角度1 ： ", angle1)

    angle2 = math.atan2(y2, x2)
    angle2 = int(angle2 * 180 / math.pi)
    # print("角度2 ： ", angle1)

    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle

    return included_angle

