from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
Image.MAX_IMAGE_PIXELS = None

if __name__ == '__main__':

    # 读取 TIFF 图片
    image_path = "D:\dataset\land-train\land-train\\rgb2id\\train\\2774_mask.png"
    image = Image.open(image_path)

    # 将图片转换为灰度图
    # gray_image = image.convert('L')

    # 将灰度图转换为 NumPy 数组
    image_array = np.array(image)

    # 计算灰度直方图
    histogram, bin_edges = np.histogram(image_array, bins=256, range=(0, 255))

    # 统计0的数量
    num_zeros = histogram[0]
    num_ones=histogram[1]
    num_twos=histogram[2]
    num_threes=histogram[3]
    num_fours=histogram[4]
    num_fives=histogram[5]
    num_sixs=histogram[6]
    num_sevens=histogram[7]
    num_255=histogram[255]

    # 计算总像素数
    total_pixels = image_array.size

    # 计算0的占比
    zero_percentage = (num_zeros / total_pixels) * 100

    print(f"Number of 0s: {num_zeros}")
    print(f"Percentage of 0s: {zero_percentage:.2f}%")

    # 显示直方图
    plt.figure(figsize=(10, 6))
    plt.plot(bin_edges[0:-1], histogram)  # bin_edges 是边界值，所以去掉最后一个元素
    plt.title('Grayscale Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Frequency')
    plt.show()
