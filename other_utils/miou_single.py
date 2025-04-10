from PIL import Image
import torch
import numpy as np



def load_image_as_array(image_path):
    """加载图像并转换为 NumPy 数组"""
    image = Image.open(image_path).convert('L')  # 将图像转换为灰度模式
    return np.array(image)


def compute_coverage(pred, label, target_class=1):
    """
    计算预测点覆盖了多少 GT 上的点
    :param pred: 预测图像的 NumPy 数组
    :param label: 真实标签图像的 NumPy 数组
    :param target_class: 目标类别值（默认为 1）
    :return: 覆盖率（预测点覆盖了多少 GT 上的点）
    """
    # 找到预测和标签中的目标点
    pred_target = (pred == target_class)
    label_target = (label == target_class)

    # 计算预测和 GT 的交集数量
    intersection = np.logical_and(pred_target, label_target).sum()
    # 计算 GT 中目标类的总点数
    gt_total = label_target.sum()

    # 计算覆盖率
    coverage = intersection / gt_total if gt_total != 0 else 0
    return coverage

def compute_acc(pred,label):
    pred_cls = (pred == 1)
    pred_sum=np.sum(pred_cls)
    # 标签中属于当前类别的像素
    label_cls = (label == 1)
    intersection = np.logical_and(pred_cls, label_cls).sum()
    acc=intersection/pred_sum
    return acc

def compute_iou(pred, label, num_classes):
    """计算 mIoU"""
    iou_list = []
    cls=1
    # for cls in range(num_classes):
    # 预测中属于当前类别的像素
    pred_cls = (pred == cls)
    # 标签中属于当前类别的像素
    label_cls = (label == cls)

    # 计算交集和并集
    intersection = np.logical_and(pred_cls, label_cls).sum()
    union = np.logical_or(pred_cls, label_cls).sum()

    # 计算 IoU（避免除零）
    iou = intersection / union if union != 0 else 0
    iou_list.append(iou)

    # 计算 mIoU
    iou = np.mean(iou_list)
    return iou

def compute_miou(pred, label, num_classes):
    iou_list = []

    for cls in range(num_classes):
        # 预测中属于当前类别的像素
        pred_cls = (pred == cls)
        # 标签中属于当前类别的像素
        label_cls = (label == cls)

        # 计算交集和并集
        intersection = np.logical_and(pred_cls, label_cls).sum()
        union = np.logical_or(pred_cls, label_cls).sum()

        # 计算 IoU（避免除零）
        iou = intersection / union if union != 0 else 0
        iou_list.append(iou)

    # 计算 mIoU，即所有类别 IoU 的平均值
    mean_iou = np.mean(iou_list)
    return mean_iou

if __name__ == '__main__':

    # 示例用法
    pred_image_path = '/root/autodl-tmp/isdnet_harr/diff_dir/diff_pred.png'  # 预测图像路径
    label_image_path = '/root/autodl-tmp/isdnet_harr/diff_dir/diff_gt.png'  # 真实标签图像路径
    # pred_image_path = "D:\dataset\\aerial\labels\\test\\austin6.tif"  # 预测图像路径
    # label_image_path = "D:\dataset\\aerial\labels\\test\\austin6.tif"  # 真实标签图像路径
    num_classes = 2  # 类别数，根据数据集进行调整


    # 读取预测和标签图像
    pred = load_image_as_array(pred_image_path)
    label = load_image_as_array(label_image_path)
    pred=pred//127
    label=label//127
    pred_count_of_ones = np.sum(pred == 1)
    pred_count_of_zeros = np.sum(pred == 0)
    # 计算 mIoU
    coverge=compute_coverage(pred,label,1)
    print(f"Coverge: {coverge}")
    iou = compute_miou(pred, label, num_classes)
    print(f"Mean ioU: {iou}")
    acc =compute_acc(pred,label)
    print(f"Acc: {acc}")
