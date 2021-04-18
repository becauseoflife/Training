import argparse


def get_global_args():
    parser = argparse.ArgumentParser()

    # 标准姿势图片的文件路径
    parser.add_argument('--picture_path', type=str, help='standard pose picture file path', default='../standard/image/')
    # 标准姿势图片识别后存放的文件路径
    parser.add_argument('--picture_store_path', type=str, help='standard pose picture file store path', default='../standard/resultImage/')

    # 识别成功保存识别图片的路径
    parser.add_argument('--identify_picture_path', type=str, help='identify pose picture store file path', default='../identify/image/')

    return parser.parse_args()