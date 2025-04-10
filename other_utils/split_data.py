import os
import shutil
from PIL import Image
import random

seed = 1
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


# if __name__ == '__main__':
#
#     # 定义文件夹路径
#     folder_path = "/root/autodl-tmp/deepglobe1/land-train/rgb2id"
#     # d_path = "/root/autodl-tmp/deepglobe2/land-train/img_dir"
#     train_folder = 'train'
#     val_folder = 'val'
#     test_folder = 'test'
#
#     # 创建保存train、val、test文件的文件夹
#     os.makedirs(os.path.join(folder_path, train_folder), exist_ok=True)
#     os.makedirs(os.path.join(folder_path, val_folder), exist_ok=True)
#     os.makedirs(os.path.join(folder_path, test_folder), exist_ok=True)
#
#     # 获取文件夹中的所有文件，并按文件名排序
#     files = os.listdir(folder_path)
#     random.shuffle(files)
#
#     # 计算划分索引
#     total_files = len(files)
#     train_end = 455
#     val_end = 662
#
#     # 将文件划分并移动到对应的文件夹
#     for index, file_name in enumerate(files):
#         # 跳过文件夹
#         if os.path.isdir(os.path.join(folder_path, file_name)):
#             continue
#
#         # 根据索引范围将文件移动到相应的文件夹
#         if index < train_end:
#             shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, train_folder, file_name))
#         elif index < val_end:
#             shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, val_folder, file_name))
#         else:
#             shutil.move(os.path.join(folder_path, file_name), os.path.join(folder_path, test_folder, file_name))



# if __name__ == '__main__':
#     convert_images_to_8bit('D:/deep_learning/ISDNet-main/deepglobe_land/rgb2id/val','D:/deep_learning/ISDNet-main/deepglobe_land/rgb2id/val1')





if __name__ == '__main__':
    # 定义图像和标签文件夹路径
    img_folder = "/root/autodl-tmp/deepglobe1/land-train/img_dir"
    label_folder = "/root/autodl-tmp/deepglobe1/land-train/rgb2id"

    # 目标子文件夹
    subsets = ['train', 'val', 'test']

    # 在 img_dir 和 rgb2id 中分别创建 train/val/test 目录
    for subset in subsets:
        os.makedirs(os.path.join(img_folder, subset), exist_ok=True)
        os.makedirs(os.path.join(label_folder, subset), exist_ok=True)

    # 获取所有图像文件，并确保标签文件也存在
    img_files = sorted([f for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))])
    label_files = sorted([f for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))])

    # 确保图像和标签数量相同
    assert len(img_files) == len(label_files), "图像与标签数量不匹配！"

    # 绑定图像和标签，保持对应关系
    paired_files = list(zip(img_files, label_files))

    # 随机打乱数据
    random.shuffle(paired_files)

    # 计算划分索引
    total = len(paired_files)
    train_end = 455 # 70% 训练集
    val_end = 662    # 20% 验证集，剩下 10% 测试集

    # 进行数据集划分
    for index, (img_name, label_name) in enumerate(paired_files):
        src_img = os.path.join(img_folder, img_name)
        src_label = os.path.join(label_folder, label_name)

        if index < train_end:
            subset = 'train'
        elif index < val_end:
            subset = 'val'
        else:
            subset = 'test'

        dst_img = os.path.join(img_folder, subset, img_name)
        dst_label = os.path.join(label_folder, subset, label_name)

        shutil.move(src_img, dst_img)
        shutil.move(src_label, dst_label)

    print("数据集划分完成！")
