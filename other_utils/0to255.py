import os
from PIL import Image
import torch
import numpy as np

def change_label_value_gpu(input_dir, output_dir):
    """
    使用 GPU 将标签图像中值为 0 的像素改为 255。

    Args:
        input_dir (str): 输入标签文件夹路径。
        output_dir (str): 输出文件夹路径。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(('.png', '.jpg', '.tif')):  # 根据文件格式过滤
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # 打开图像并转为灰度图像
            img = Image.open(input_path).convert('L')

            # 将图像数据转为 PyTorch 张量并移动到 GPU
            img_tensor = torch.tensor(np.array(img), device='cuda', dtype=torch.uint8)

            # 修改像素值 (值为 0 的替换为 255)
            img_tensor = torch.where(img_tensor == 0, torch.tensor(255, device='cuda', dtype=torch.uint8), img_tensor)

            # 将修改后的张量移回 CPU，并转换为图像格式
            modified_img = Image.fromarray(img_tensor.cpu().numpy())
            modified_img.save(output_path)

            print(f"Processed: {file_name}")

if __name__ == '__main__':

    # 输入和输出目录路径
    input_directory = "/root/autodl-tmp/land-train/land-train/rgb2id/val"  # 替换为标签图像文件夹路径
    output_directory = "/root/autodl-tmp/land-train/land-train/rgb2id_t/cal"  # 替换为输出文件夹路径

    change_label_value_gpu(input_directory, output_directory)
