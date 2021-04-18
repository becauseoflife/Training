# -*- coding: utf-8 -*-
import copy
import argparse
import os

import cv2 as cv
import mediapipe as mp

from args import get_pose_model_args, get_global_args
from utils import LoadData
from utils import calc_bounding_rect, draw_landmarks, draw_bounding_rect
from utils import calc_angle

from multiprocessing import Process, Queue


def get_picture_angle(file, index):

    # 参数分析v #################################################################
    # 识别模型参数
    args = get_pose_model_args()
    upper_body_only = args.upper_body_only
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = args.use_brect
    # 存储路径参数
    # 获取参数
    args = get_global_args()
    picture_store_path = args.picture_store_path

    # 模型加载 #############################################################
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        upper_body_only=upper_body_only,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # 图片读取 #####################################################
    image = cv.imread(file)
    # 镜像显示
    image = cv.flip(image, 1)
    debug_image = copy.deepcopy(image)

    # 检测实施 #############################################################
    # 由于OpenCV默认的颜色空间是BGR，但是一般我们说的颜色空间为RGB，这里 mediapipe 便修改了颜色空间
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = pose.process(image)
    angle_json = None
    # 绘画 ################################################################
    if results.pose_landmarks is not None:
        # 外接矩形的计算
        brect = calc_bounding_rect(debug_image, results.pose_landmarks)
        # 绘画
        debug_image, landmark_points = draw_landmarks(debug_image, results.pose_landmarks)
        debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        # 计算角度
        angle_json = calc_angle(landmark_points)

    # 屏幕反射 #############################################################
    # while True:
    #     cv.imshow('MediaPipe Pose Demo', debug_image)
    #     # 按键处理（ESC：结束） #################################################
    #     cv.imwrite('../resultVideo/res.png', debug_image)
    #     key = cv.waitKey(1)
    #     if key == 27:  # ESC
    #         break

    # 获取文件路径的后缀
    suffix = os.path.splitext(file)[-1]
    # 角标准图片识别结果保存
    cv.imwrite(picture_store_path + str(index) + suffix, debug_image)

    data = {'data': angle_json}
    # loadJson = LoadData('angle')
    # print("长度: ", len(data))
    # print("data: ", data)
    # loadJson.save_data(data)

    cv.destroyAllWindows()

    return data, debug_image


# get_picture_angle('../standard/image/test1.jpg')