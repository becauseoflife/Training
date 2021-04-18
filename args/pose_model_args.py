import argparse

# argparse 模块可以让人轻松编写用户友好的命令行接口。
# 程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
# argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
# 此模块是 Python 标准库中推荐的命令行解析模块。


def get_pose_model_args():
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