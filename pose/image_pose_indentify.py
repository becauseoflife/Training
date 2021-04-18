#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import datetime
import math
import os

import cv2 as cv
import numpy as np
import mediapipe as mp
from PIL import Image

from args import get_pose_model_args, get_global_args
from utils import CvFpsCalc, LoadData, getUrlFile
from utils import interval
from pose.image_pose_data import get_picture_angle
from utils import calc_bounding_rect, draw_landmarks, draw_bounding_rect
from utils import get_log_data
from utils import calc_angle

import globalVariable.global_variable as global_dict


# 传入标准的姿势图片数组
def image_pose_indentify():
    # 获取 GUI界面 全局变量
    trainingGUI = global_dict.get_value('trainingGUI')

    # 参数分析v #################################################################
    # 模型的参数
    pose_args = get_pose_model_args()
    cap_device = pose_args.device
    cap_width = pose_args.width
    cap_height = pose_args.height
    upper_body_only = pose_args.upper_body_only
    min_detection_confidence = pose_args.min_detection_confidence
    min_tracking_confidence = pose_args.min_tracking_confidence
    use_brect = pose_args.use_brect
    # 存储路径
    path_args = get_global_args()
    identify_picture_path = path_args.identify_picture_path
    picture_path = path_args.picture_path

    # 获取标准动作图片文件夹下的所有图片===================================
    array_image = []
    for item in getUrlFile(picture_path):
        array_image.append(item)
        # print(item)
    # trainingGUI.update_image("../standard/image/test3.png")
    # 没有标准图片
    if len(array_image) == 0:
        trainingGUI.set_text("项目路径 standard/image/ 下没有标准图片")
        return

    # 照相机准备 ###############################################################
    #  VideoCapture()中参数是0，表示打开笔记本的内置摄像头；
    # 参数是1，则打开外置摄像头；
    # 其他数字则代表其他设备；
    # 参数是视频文件的路径则打开指定路径下的视频。
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # 设置全局变量
    global_dict.set_value('cap', cap)

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

    # 保存视频指定编码器
    # fourcc = cv.VideoWriter_fourcc(*'XVID')
    # video_writer = None
    # flag = True

    # print("要处理的突变数组长度: ", len(array_image))
    # print("参数: ", array_image)
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
            # trainingGUI.set_text("没有默认的摄像设备使用···")
            return
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
            data_arr, standard_image = get_picture_angle(file=array_image[index][0], index=index + 1)
            # 显示标准姿势识别的图片
            trainingGUI.update_standard_image(standard_image)
            # 打印日志
            log = get_log_data('标准的角度', data_arr['data'], index + 1)
            trainingGUI.add_log(log)

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
                # 打印日志
                log = get_log_data('识别的角度', angle_json, index + 1)
                trainingGUI.add_log_empty(log)
                # 文件后缀
                suffix = os.path.splitext(array_image[index][0])[-1]
                # 保存识别的图片
                cv.imwrite(identify_picture_path + str(index + 1) + suffix, debug_image)
                print("识别成功！==============", index + 1, "===============")
                print("识别的角度： ", angle_json)
                print("标准的角度： ", data_arr['data'])
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

        # 位置参数说明： 图片 要添加的文字 文字添加到图片上的位置 字体的类型 字体大小 字体颜色 字体粗细
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

    # 重置画面
    trainingGUI.identify_compass()
    # 显示训练完成提示框
    trainingGUI.show_info()
    # video_writer.release()
    cap.release()
    cv.destroyAllWindows()


# 比较两个姿势角度数据是否相等
def compareAngle(standard_angle, angle_json, mistake=20, hip_mistake=20):
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
        if standard != -1 and angle not in interval(standard, hip_mistake):
            if angle < standard - hip_mistake:
                tipDecText_l.append("请增大左侧腰部和左大腿之间的幅度！")
            if angle > standard + hip_mistake:
                tipDecText_l.append("请减小左侧腰部和左大腿之间的幅度！")
    elif angle == -1 and standard != -1:
        tipDecText_l.append("未识别左腰部")

    standard = standard_angle['hip']['right_hip_angle']
    angle = angle_json['hip']['right_hip_angle']
    if angle != -1:
        if standard != -1 and angle not in interval(standard, hip_mistake):
            if angle < standard - hip_mistake:
                tipDecText_r.append("请增大右侧腰部和右大腿之间的幅度！")
            if angle > standard + hip_mistake:
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
