from PIL import Image, ImageChops
import numpy as np


def p_g_difference(predict_img, ground_truth_img):
    # 计算两张图片的差异
    diff = predict_img - ground_truth_img
    diff=np.abs(diff)
    # 将差异转换为图像显示
    diff_img = Image.fromarray(diff.astype('uint8'))
    diff_img.save("diff_numpy.png")

if __name__ == '__main__':
    # 打开图片
    predict_img = np.array(Image.open("D:\deep_learning\ISDNet-main\show_dir\hjd_9_2.png").convert('L'),dtype=np.int16)
    ground_truth_img =np.array(Image.open("D:\\temp\hjd_9_2.png"),dtype=np.int16)
    # predict_img = np.array(Image.open('D:\deep_learning\ISDNet-main\show_dir\\vienna3.tif').convert('L'))
    # ground_truth_img =np.array(Image.open('D:\dataset\\NEW2-AerialImageDataset\AerialImageDataset\\train\gt\\vienna3.tif'))
    p_g_difference(predict_img,ground_truth_img)
    # p_g_difference(ground_truth_img,predict_img)


