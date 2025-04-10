import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from other_utils.map_label import reverse_map_labels


def save_heatmap(feature_map_np, save_dir='D:\deep_learning\ISDNet-main\heatmap', filename="heatmap.png", channel=0,ignore_value=255):
    # 选择一个通道进行可视化
    heatmap = feature_map_np[channel]

    # 创建保存文件夹（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 生成热力图
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()

    # 生成保存路径
    save_path = os.path.join(save_dir, filename)

    # 保存热力图到指定文件夹，不显示
    plt.savefig(save_path)
    plt.close()  # 关闭图像防止内存泄漏

    print(f"Heatmap saved at {save_path}")


def save_heatmap_avg(feature_map_np, save_dir='D:\deep_learning\ISDNet-main\heatmap', filename="heatmap_avg.png"):
    # 计算所有通道的平均值
    heatmap_avg = np.mean(feature_map_np, axis=0)

    # 创建保存文件夹（如果不存在）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 生成热力图
    plt.imshow(heatmap_avg, cmap='hot', interpolation='nearest')
    plt.colorbar()

    # 生成保存路径
    save_path = os.path.join(save_dir, filename)

    # 保存热力图到指定文件夹，不显示
    plt.savefig(save_path)
    plt.close()

    print(f"Average heatmap saved at {save_path}")


def save_image(data,filename,save_dir='D:\deep_learning\ISDNet-main\heatmap',ignore_value=255):
    # data=data.squeeze(0)
    # data[data == ignore_value] = 0
    # data=data*255/2
    data[data==1]=127

    data = data.astype(np.uint8)
    image = Image.fromarray(data, mode='L')
    save_path = os.path.join(save_dir, filename)
    image.save(save_path+'.png')

def save_rgb_image(data,filename,save_dir='D:\deep_learning\ISDNet-main\heatmap'):
    data = data.astype(np.uint8)
    data=reverse_map_labels(data)
    image = Image.fromarray(data)
    save_path = os.path.join(save_dir, filename)
    image.save(save_path+'.png')



