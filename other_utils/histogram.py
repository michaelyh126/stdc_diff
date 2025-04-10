import torch
import numpy as np
import matplotlib.pyplot as plt


def tensor_histogram(tensor):
    # 如果tensor在GPU上，将其移动到CPU
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # 将tensor转换为numpy数组
    img_np = tensor.numpy()
    # 如果输入是四维张量（如批量图像），取第一个图像
    if len(img_np.shape) == 4:
        img_np = img_np[0]
    # 转换通道顺序从CHW到HWC
    img_np = np.transpose(img_np, (1, 2, 0))

    # 定义颜色列表，对应不同通道
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    num_channels = img_np.shape[2]

    for i in range(num_channels):
        # 获取当前通道的数据
        channel_data = img_np[:, :, i].flatten()
        # 计算当前通道数据的最小值和最大值
        min_val = np.min(channel_data)
        max_val = np.max(channel_data)
        # 计算当前通道的像素直方图，范围设置为该通道的最值
        hist, bins = np.histogram(channel_data, bins=256, range=[min_val, max_val])
        # 绘制当前通道的直方图
        plt.plot(hist, color=colors[i % len(colors)])

    # 设置图表标题和坐标轴标签
    plt.title('7通道像素直方图')
    plt.xlabel('像素值')
    plt.ylabel('像素数量')

    # 显示图表
    plt.show()

