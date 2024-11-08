import numpy as np
from PIL import Image
import os

def compute_iou(pred, gt, num_classes):
    iou_per_class = []
    for cls in range(num_classes):
        pred_class = (pred == cls)
        gt_class = (gt == cls)

        intersection = np.logical_and(pred_class, gt_class).sum()
        union = np.logical_or(pred_class, gt_class).sum()

        if union == 0:
            iou = float('nan')  # 如果该类不存在，则不计算IoU
        else:
            iou = intersection / union
        iou_per_class.append(iou)

    # 计算有效类别的平均IoU
    valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
    miou = np.mean(valid_ious)
    return miou, iou_per_class


def calculate_mean_miou(pred_dir, gt_dir, num_classes):
    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    miou_list = []

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        # 读取预测和真实标签图片
        pred_image = Image.open(pred_path)
        pred_image = pred_image.convert(mode='L')
        gt_image = Image.open(gt_path)

        # 将图片转换为NumPy数组
        pred_array = np.array(pred_image)
        gt_array = np.array(gt_image)

        # 计算 mIoU
        miou,iou_per_class  = compute_iou(pred_array//128, gt_array//128, num_classes)
        miou_list.append(miou)

    # 计算所有图片对的 mIoU 平均值
    mean_miou = np.mean(miou_list)
    return mean_miou

if __name__ == '__main__':

    num_classes = 2
    # 读取预测和真实标签图片
    pred_dir = 'D:\deep_learning\ISDNet-main\show_dir'
    gt_dir = 'D:\deep_learning\ISDNet-main\\root_path\labels\\test'

    mean_miou = calculate_mean_miou(pred_dir, gt_dir, num_classes)
    print(f"mIoU: {mean_miou}")

