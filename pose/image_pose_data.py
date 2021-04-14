# -*- coding: utf-8 -*-
import copy
import argparse
import cv2 as cv
import mediapipe as mp

from utils import LoadData
from utils import calc_bounding_rect, draw_landmarks, draw_bounding_rect
from utils import calc_angle

from multiprocessing import Process, Queue


# argparse 模块可以让人轻松编写用户友好的命令行接口。
# 程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
# argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
# 此模块是 Python 标准库中推荐的命令行解析模块。
def get_args():
    parser = argparse.ArgumentParser()

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


def get_picture_angle(file):

    # 参数分析v #################################################################
    args = get_args()

    upper_body_only = args.upper_body_only
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

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

    # queue.put(debug_image)

    # print("angle_json")
    # print(angle_json)
    data = {'data': angle_json}
    # loadJson = LoadData('angle')
    # print("长度: ", len(data))
    # print("data: ", data)
    # loadJson.save_data(data)

    cv.destroyAllWindows()

    return data, debug_image


# get_picture_angle('../standard/image/test1.jpg')