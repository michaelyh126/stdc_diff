import os
from PIL import Image

if __name__ == '__main__':

    # 指定文件夹路径
    folder_path = "D:\deep_learning\ISDNet-main\dataset_island512\imgs"

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)

            # 打开32位的PNG文件
            image = Image.open(file_path)

            # 将图片转换为RGB模式，RGB模式是24位
            rgb_image = image.convert("RGB")

            # 保存为24位的PNG文件
            rgb_image.save(file_path)  # 覆盖原文件

    print("所有PNG文件已成功转换为24位。")

