import os
from PIL import Image
import numpy as np

def modify_pixels_in_folder(input_folder, output_folder):
    try:
        # 检查输出文件夹是否存在，如果不存在则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历输入文件夹中的所有图像文件
        for filename in os.listdir(input_folder):
            if filename.endswith(('.png', '.tif')):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)

                # 读取TIFF格式的图像
                image = Image.open(input_path)

                # 将像素值为1的像素设置为255
                image = image.point(lambda x: x * 255)


                # 保存修改后的图像到输出文件夹
                Image.fromarray(np.uint8(image)).save(output_path)

                print(f"Pixels modified in {filename} and saved to {output_path}")

    except Exception as e:
        print(f"Error: {e}")



if __name__ == '__main__':
    # 指定输入文件夹路径
    input_folder_path = 'D:\sea\cropped_labels'
    # input_folder_path = 'D:\dataset\VOCdevkit_512_edge\VOC2007\SegmentationClass'

    # 指定输出文件夹路径
    output_folder_path = 'D:\sea\\test_labels'
    # output_folder_path = 'D:\deep_learning\ISDNet-main\dataset_island512\labels'
    # 调用函数处理文件夹中的所有图像
    modify_pixels_in_folder(input_folder_path, output_folder_path)
