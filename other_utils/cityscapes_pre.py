import os
from PIL import Image
import numpy as np

# # 定义 34 类到 19 类的映射表
# CITYSCAPES_34_TO_19 = {
#     0: 255, 1: 255, 2: 255, 3: 255, 4: 255,  # 忽略的类别
#     5: 0, 6: 1, 7: 2, 8: 3, 9: 4,           # 道路、人行道、建筑物、墙、篱笆
#     10: 5, 11: 6, 12: 7, 13: 8, 14: 9,      # 电线杆、红绿
#     # 灯、交通标志、植被、地形
#     15: 10, 16: 11, 17: 12, 18: 13, 19: 14, # 天空、行人、骑车人、汽车、卡车
#     20: 15, 21: 16, 22: 17, 23: 18,         # 公交车、火车、摩托车、自行车
#     24: 255, 25: 255, 26: 255, 27: 255,     # 忽略的类别
#     28: 255, 29: 255, 30: 255, 31: 255,
#     32: 255, 33: 255
# }
ignore_label=255
CITYSCAPES_34_TO_19= {-1: ignore_label, 0: ignore_label,
                      1: ignore_label, 2: ignore_label,
                      3: ignore_label, 4: ignore_label,
                      5: ignore_label, 6: ignore_label,
                      7: 0, 8: 1, 9: ignore_label,
                      10: ignore_label, 11: 2, 12: 3,
                      13: 4, 14: ignore_label, 15: ignore_label,
                      16: ignore_label, 17: 5, 18: ignore_label,
                      19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                      25: 12, 26: 13, 27: 14, 28: 15,
                      29: ignore_label, 30: ignore_label,
                      31: 16, 32: 17, 33: 18}


# 标签映射函数
def map_cityscapes_label(label):
    """
    将 34 类标签映射为 19 类
    参数:
        label: np.ndarray, 标签数组 (H, W)
    返回:
        np.ndarray, 映射后的标签数组 (H, W)
    """
    mapped_label = label.copy()
    for original_id, train_id in CITYSCAPES_34_TO_19.items():
        mapped_label[label == original_id] = train_id
    return mapped_label

# 原地修改文件的函数
def process_and_overwrite(input_path):
    """
    原地处理标签文件，将其从 34 类转换为 19 类。
    参数:
        input_path: str, 标签文件路径
    """
    label = np.array(Image.open(input_path))
    mapped_label = map_cityscapes_label(label)
    mapped_image = Image.fromarray(mapped_label.astype(np.uint8))
    mapped_image.save(input_path)  # 覆盖原始文件
    print(f"Modified: {input_path}")

# 遍历 Cityscapes 标签文件夹并修改
def modify_cityscapes_labels(input_dir):
    """
    遍历 Cityscapes 数据集目录，原地修改标签文件。
    参数:
        input_dir: str, Cityscapes gtFine 文件夹路径
    """
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_labelIds.png"):  # 只处理标签文件
                file_path = os.path.join(root, file)
                process_and_overwrite(file_path)

if __name__ == '__main__':

    # 指定 Cityscapes gtFine 文件夹路径
    cityscapes_gtfine_dir = "/root/autodl-tmp/cityscapes/gtFine/val"
    modify_cityscapes_labels(cityscapes_gtfine_dir)
