from PIL import Image
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def random_crop(image, label, crop_size, max_ratio=0.75):
    img_width, img_height = image.size
    crop_width, crop_height = crop_size

    if img_width < crop_width or img_height < crop_height:
        raise ValueError("Crop size must be smaller than the image size.")

    while True:
        # 随机选择裁剪的左上角坐标
        left = random.randint(0, img_width - crop_width)
        top = random.randint(0, img_height - crop_height)

        # 计算右下角坐标
        right = left + crop_width
        bottom = top + crop_height

        # 裁剪图像
        cropped_image = image.crop((left, top, right, bottom))

        # 裁剪标签
        # cropped_label = label[top:bottom, left:right]
        cropped_label = label.crop((left, top, right, bottom))

        # 检测裁剪后的标签中目标像素值的比例
        total_pixels = cropped_label.size[0]*cropped_label.size[1]
        white_num = np.sum(cropped_label == 255)
        black_num=total_pixels-white_num
        white_ratio = white_num / total_pixels
        black_ratio = black_num / total_pixels

        if white_ratio <= max_ratio :
            if black_ratio <= max_ratio :
                return cropped_image, cropped_label, True
            else:
                continue
        else:
            # 如果目标像素值比例过高，重新裁剪
            continue


if __name__ == '__main__':
    image = Image.open('D:/dataset/SDisland/clip_hjd1.tif')
    label = Image.open('D:/dataset/SDisland/clip_hjd_mask3.tif')

    crop_size = (5000, 5000)  # 定义裁剪的大小

    cropped_image, cropped_label, valid = random_crop(image, label, crop_size)

    if valid:
        # 保存裁剪后的结果
        cropped_image.save('cropped_image.jpg')
        cropped_label.save('cropped_label.png')
    else:
        print("裁剪后的标签未满足条件，请调整策略。")
