# from PIL import Image
# import os
#
# if __name__ == '__main__':
#
#     # 指定要处理的文件夹路径
#     folder_path = "D:\deep_learning\ISDNet-main\\root_path\labels\\test"
#
#     # 遍历文件夹中的所有图片
#     for filename in os.listdir(folder_path):
#         if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp','tif','tiff')):  # 只处理图片文件
#             file_path = os.path.join(folder_path, filename)
#
#             # 打开图片
#             image = Image.open(file_path)
#
#             # 将图片转换为灰度模式
#             image = image.convert("L")
#
#             # 获取图片的像素数据
#             pixels = image.load()
#
#             # 遍历每个像素
#             for i in range(image.width):
#                 for j in range(image.height):
#                     if pixels[i, j] == 255:
#                         pixels[i, j] = 1  # 将像素值为255的像素改为1
#
#             # 保存修改后的图片（原地覆盖）
#             image.save(file_path)
#
#     print("图片已原地修改完成！")

import os
from PIL import Image
import torch
import torchvision.transforms as transforms

if __name__ == '__main__':

    # 指定要处理的文件夹路径
    folder_path = "/root/autodl-tmp/Levir/labels/val"

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义图片转换操作
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    # 遍历文件夹中的所有图片
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp','tif')):  # 只处理图片文件
            file_path = os.path.join(folder_path, filename)

            # 打开图片并转换为张量
            image = Image.open(file_path).convert("L")
            image_tensor = to_tensor(image).to(device)

            # 找到值为255的像素并替换为1
            image_tensor[image_tensor == 1.0] = 1.0 / 255.0  # PyTorch tensor中像素值范围为[0,1]

            # 将张量转换回图片并保存
            modified_image = to_pil(image_tensor.cpu())
            modified_image.save(file_path)

    print("图片已原地修改完成！")
