import os
import shutil
from PIL import Image
import random

seed = 0
random.seed(seed)



def convert_images_to_8bit(input_folder, output_folder, mode='L'):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 构建完整的文件路径
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 打开图像
            with Image.open(input_path) as img:
                # 转换为8位图像
                img_8bit = img.convert(mode)

                # 保存8位图像到输出文件夹
                img_8bit.save(output_path)
                print(f"Converted {filename} to 8-bit and saved to {output_path}")

# if __name__ == '__main__':
#
#     # 定义文件夹路径
#     folder_path = './deepglobe_land/land-train'
#     odd_folder = 'odd'
#     even_folder = 'even'
#
#     # 创建保存奇数和偶数文件的文件夹
#     os.makedirs(os.path.join(folder_path, odd_folder), exist_ok=True)
#     os.makedirs(os.path.join(folder_path, even_folder), exist_ok=True)
#
#     # 获取文件夹中的所有文件，并按文件名排序
#     files = sorted(os.listdir(folder_path))
#         # 遍历文件并根据其顺序的奇偶性进行分类
#     for index, file_name in enumerate(files):
#         # 跳过文件夹
#         if os.path.isdir(os.path.join(folder_path, file_name)):
#             continue
#
#         # 根据文件的顺序将文件移动到相应的文件夹
#         if index % 2 == 0:
#             shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, even_folder, file_name))
#         else:
#             shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, odd_folder, file_name))


import os
import shutil


if __name__ == '__main__':

    # 定义文件夹路径
    folder_path = "D:\dataset\\aerial\\imgs"
    train_folder = 'train'
    val_folder = 'val'
    test_folder = 'test'

    # 创建保存train、val、test文件的文件夹
    os.makedirs(os.path.join(folder_path, train_folder), exist_ok=True)
    os.makedirs(os.path.join(folder_path, val_folder), exist_ok=True)
    os.makedirs(os.path.join(folder_path, test_folder), exist_ok=True)

    # 获取文件夹中的所有文件，并按文件名排序
    files = os.listdir(folder_path)
    random.shuffle(files)

    # 计算划分索引
    total_files = len(files)
    train_end = int(0.75 * total_files)
    val_end = int(0.9 * total_files)

    # 将文件划分并移动到对应的文件夹
    for index, file_name in enumerate(files):
        # 跳过文件夹
        if os.path.isdir(os.path.join(folder_path, file_name)):
            continue

        # 根据索引范围将文件移动到相应的文件夹
        if index < train_end:
            shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, train_folder, file_name))
        elif index < val_end:
            shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, val_folder, file_name))
        else:
            shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, test_folder, file_name))



# if __name__ == '__main__':
#     convert_images_to_8bit('D:/deep_learning/ISDNet-main/deepglobe_land/rgb2id/val','D:/deep_learning/ISDNet-main/deepglobe_land/rgb2id/val1')
