import os
import numpy as np
import torch
from PIL import Image

# 定义颜色到类别的映射
color_map = {
    (0, 255, 255): 1,  # 城市
    (255, 255, 0): 2,  # 农田
    (255, 0, 255): 3,  # 牧场
    (0, 255, 0): 4,  # 森林
    (0, 0, 255): 5,  # 水体
    (255, 255, 255): 6,  # 荒地
    (0, 0, 0): 0  # 未知
}

# color_map = {
#     (64, 128, 64): 1,  # Animal
#     (192, 0, 128): 2,  # Archway
#     (0, 128, 192): 3,  # Bicyclist
#     (0, 128, 64): 4,  # Bridge
#     (128, 0, 0): 5,  # Building
#     (64, 0, 128): 6,  # Car
#     (64, 0, 192): 7,  # CartLuggagePram
#     (192, 128, 64): 8,  # Child
#     (192, 192, 128): 9,  # Column_Pole
#     (64, 64, 128): 10,  # Fence
#     (128, 0, 192): 11,  # LaneMkgsDriv
#     (192, 0, 64): 12,  # LaneMkgsNonDriv
#     (128, 128, 64): 13,  # Misc_Text
#     (192, 0, 192): 14,  # MotorcycleScooter
#     (128, 64, 64): 15,  # OtherMoving
#     (64, 192, 128): 16,  # ParkingBlock
#     (64, 64, 0): 17,  # Pedestrian
#     (128, 64, 128): 18,  # Road
#     (128, 128, 192): 19,  # RoadShoulder
#     (0, 0, 192): 20,  # Sidewalk
#     (192, 128, 128): 21,  # SignSymbol
#     (128, 128, 128): 22,  # Sky
#     (64, 128, 192): 23,  # SUVPickupTruck
#     (0, 0, 64): 24,  # TrafficCone
#     (0, 64, 64): 25,  # TrafficLight
#     (192, 64, 128): 26,  # Train
#     (128, 128, 0): 27,  # Tree
#     (192, 128, 192): 28,  # Truck_Bus
#     (64, 0, 64): 29,  # Tunnel
#     (192, 192, 0): 30,  # VegetationMisc
#     (0, 0, 0): 0,  # Void
#     (64, 192, 0): 31,  # Wall
# }



# 将颜色映射转换为张量
def create_color_tensor(color_map):
    colors = torch.tensor(list(color_map.keys()), dtype=torch.uint8).cuda()
    classes = torch.tensor(list(color_map.values()), dtype=torch.uint8).cuda()
    return colors, classes


def map_labels_cuda(label_image):
    # 转换为 PyTorch 张量并移动到 GPU
    label_image_tensor = torch.tensor(label_image, dtype=torch.uint8).cuda()
    label_array = torch.zeros((label_image_tensor.shape[0], label_image_tensor.shape[1]), dtype=torch.uint8).cuda()

    colors, classes = create_color_tensor(color_map)

    for color, class_index in zip(colors, classes):
        mask = (label_image_tensor == color).all(dim=-1)
        label_array[mask] = class_index

    return label_array.cpu().numpy()  # 转回 CPU 并转换为 NumPy 数组


# 反向映射函数
def reverse_map_labels(label_array):
    reverse_color_map = {v: k for k, v in color_map.items()}
    colored_image = np.zeros((*label_array.shape, 3), dtype=np.uint8)

    for class_index, color in reverse_color_map.items():
        colored_image[label_array == class_index] = color

    return colored_image


# if __name__ == '__main__':
#
#     # 指定包含标签图像的文件夹路径
#     folder_path = 'D:\dataset\camvid\labels\\test'
#
#     # 遍历文件夹中的所有图像
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.png') or filename.endswith('.jpg'):
#             image_path = os.path.join(folder_path, filename)
#
#             # 加载标签图像
#             label_image = Image.open(image_path).convert('RGB')
#
#             # 进行标签映射
#             label_array = map_labels_cuda(np.array(label_image))
#
#             # 保存映射后的标签到原始位置
#             mapped_image = Image.fromarray(label_array)
#             mapped_image.save(image_path)  # 保存到原始路径
#
#             print(f"Processed and saved: {filename}")


if __name__ == '__main__':
    # 指定包含标签图像的文件夹路径
    folder_path = 'D:\dataset\land-train\land-train\\rgb2id\\test'

    # 遍历文件夹中的所有图像
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)

            # 加载映射后的标签图像
            label_image = Image.open(image_path)
            label_array = np.array(label_image)

            # 还原颜色图像
            colored_image = reverse_map_labels(label_array)

            # 保存还原后的图像到原始位置或另存为新文件
            restored_image = Image.fromarray(colored_image)
            restored_image.save(os.path.join(folder_path, f'restored_{filename}'))  # 新文件名

            print(f"Restored and saved: restored_{filename}")
