import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging
Image.MAX_IMAGE_PIXELS = None

# 配置日志
logging.basicConfig(
    filename='image_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

Image.MAX_IMAGE_PIXELS = None


def create_dir(path):
    os.makedirs(path, exist_ok=True)


def get_largest_class_ratio(mask, num_classes):
    """
    计算标签中最大类别所占比例，忽略255值
    """
    mask_array = np.array(mask)

    # 过滤掉255值
    valid_mask = mask_array != 255
    valid_pixels = mask_array[valid_mask]

    if valid_pixels.size == 0:
        return 0  # 如果没有有效像素，返回0

    counts = np.bincount(valid_pixels.flatten(), minlength=num_classes)

    # 处理可能没有出现的类别
    if len(counts) < num_classes:
        counts = np.pad(counts, (0, num_classes - len(counts)), 'constant')

    max_count = counts.max()
    total = valid_pixels.size
    ratio = max_count / total
    return ratio


def crop_image_and_mask(image, mask, crop_size, stride, num_classes, max_ratio=0.8):
    """
    对图像和掩码进行裁剪，并筛选符合条件的裁剪区域
    """
    img_width, img_height = image.size
    crops = []
    mask_crops = []

    for y in range(0, img_height - crop_size + 1, stride):
        for x in range(0, img_width - crop_size + 1, stride):
            try:
                img_crop = image.crop((x, y, x + crop_size, y + crop_size))
                mask_crop = mask.crop((x, y, x + crop_size, y + crop_size))

                ratio = get_largest_class_ratio(mask_crop, num_classes)
                if ratio <= max_ratio:
                    crops.append(img_crop)
                    mask_crops.append(mask_crop)
            except Exception as e:
                logging.error(f"Error cropping at ({x}, {y}) in image: {e}")
                continue

    return crops, mask_crops


def process_dataset(images_dir, masks_dir, output_images_dir, output_masks_dir, crop_size, stride, num_classes,
                    max_ratio=0.8):
    """
    处理整个数据集，裁剪并保存符合条件的图像和掩码
    """
    create_dir(output_images_dir)
    create_dir(output_masks_dir)

    image_names = sorted(os.listdir(images_dir))
    mask_names = sorted(os.listdir(masks_dir))

    if len(image_names) != len(mask_names):
        logging.error("图像和掩码的数量不匹配。")
        raise ValueError("图像和掩码的数量不一致")

    for img_name, mask_name in tqdm(zip(image_names, mask_names), total=len(image_names), desc="处理图像"):
        img_path = os.path.join(images_dir, img_name)
        mask_path = os.path.join(masks_dir, mask_name)

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")  # 假设标签是灰度图
        except Exception as e:
            logging.error(f"打开文件失败：{img_name} 或 {mask_name}。错误：{e}")
            continue

        img_crops, mask_crops = crop_image_and_mask(image, mask, crop_size, stride, num_classes, max_ratio)

        for idx, (img_crop, mask_crop) in enumerate(zip(img_crops, mask_crops)):
            base_name = os.path.splitext(img_name)[0]
            crop_img_name = f"{base_name}_crop_{idx}.png"
            crop_mask_name = f"{os.path.splitext(mask_name)[0]}_crop_{idx}.png"

            try:
                img_crop.save(os.path.join(output_images_dir, crop_img_name))
                mask_crop.save(os.path.join(output_masks_dir, crop_mask_name))
            except Exception as e:
                logging.error(f"保存裁剪图像失败：{crop_img_name} 或 {crop_mask_name}。错误：{e}")
                continue


if __name__ == "__main__":
    # 参数设置
    ORIGINAL_IMAGES_DIR = r'D:\sea\added_imgs'  # 原图文件夹路径
    ORIGINAL_MASKS_DIR = r'D:\sea\added_labels'  # 标签文件夹路径
    OUTPUT_IMAGES_DIR = r'D:\sea\cropped_imgs'  # 裁剪后图像保存路径
    OUTPUT_MASKS_DIR = r'D:\sea\cropped_labels'  # 裁剪后标签保存路径
    CROP_SIZE = 5000  # 裁剪大小 n x n
    STRIDE = 2500  # 步长，控制重叠程度
    NUM_CLASSES = 2  # 类别数量，0和1
    MAX_RATIO = 0.8  # 最大类别比例

    try:
        process_dataset(
            images_dir=ORIGINAL_IMAGES_DIR,
            masks_dir=ORIGINAL_MASKS_DIR,
            output_images_dir=OUTPUT_IMAGES_DIR,
            output_masks_dir=OUTPUT_MASKS_DIR,
            crop_size=CROP_SIZE,
            stride=STRIDE,
            num_classes=NUM_CLASSES,
            max_ratio=MAX_RATIO
        )
        logging.info("数据集处理成功完成。")
    except Exception as e:
        logging.critical(f"数据集处理失败：{e}")

# import os
# import torch
# import torchvision.transforms as T
# from PIL import Image
# import numpy as np
# from tqdm import tqdm
# Image.MAX_IMAGE_PIXELS = None
#
#
# def create_dir(path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#
# def get_largest_class_ratio(mask_tensor, num_classes):
#     """
#     计算标签中最大类别所占比例
#     """
#     # mask_tensor: (1, H, W)
#     counts = torch.bincount(mask_tensor.view(-1), minlength=num_classes).float()
#     max_count = counts.max()
#     total = mask_tensor.numel()
#     ratio = max_count / total
#     return ratio
#
#
# def crop_image_and_mask(image_tensor, mask_tensor, crop_size, stride, num_classes, max_ratio=0.8):
#     """
#     对图像和掩码进行裁剪，并筛选符合条件的裁剪区域
#     """
#     _, img_height, img_width = image_tensor.shape
#     crops = []
#     mask_crops = []
#
#     # 计算裁剪的起始点
#     y_steps = list(range(0, img_height - crop_size + 1, stride))
#     x_steps = list(range(0, img_width - crop_size + 1, stride))
#
#     for y in y_steps:
#         for x in x_steps:
#             img_crop = image_tensor[:, y:y + crop_size, x:x + crop_size]
#             mask_crop = mask_tensor[:, y:y + crop_size, x:x + crop_size]
#
#             ratio = get_largest_class_ratio(mask_crop, num_classes)
#             if ratio <= max_ratio:
#                 crops.append(img_crop)
#                 mask_crops.append(mask_crop)
#
#     return crops, mask_crops
#
#
# def process_dataset(
#         images_dir, masks_dir, output_images_dir, output_masks_dir,
#         crop_size, stride, num_classes, max_ratio=0.8, device='cuda'
# ):
#     """
#     处理整个数据集，裁剪并保存符合条件的图像和掩码
#     """
#     create_dir(output_images_dir)
#     create_dir(output_masks_dir)
#
#     image_names = sorted(os.listdir(images_dir))
#     mask_names = sorted(os.listdir(masks_dir))
#
#     assert len(image_names) == len(mask_names), "图像和掩码的数量不一致"
#
#     # 定义转换
#     transform = T.Compose([
#         T.ToTensor(),
#     ])
#
#     for img_name, mask_name in tqdm(zip(image_names, mask_names), total=len(image_names), desc="处理图像"):
#         img_path = os.path.join(images_dir, img_name)
#         mask_path = os.path.join(masks_dir, mask_name)
#
#         # 加载图像和掩码
#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")  # 假设标签是灰度图
#
#         image_tensor = transform(image).to(device)  # (3, H, W)
#         mask_tensor = transform(mask).to(device) * 255  # 转换为 0-255 的类别编号
#         mask_tensor = mask_tensor.long().squeeze(0)  # (H, W)
#         mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)
#
#         # 裁剪并筛选
#         img_crops, mask_crops = crop_image_and_mask(
#             image_tensor, mask_tensor, crop_size, stride, num_classes, max_ratio
#         )
#
#         for idx, (img_crop, mask_crop) in enumerate(zip(img_crops, mask_crops)):
#             # 移动到 CPU 并转换为 PIL 图像
#             img_crop_cpu = img_crop.cpu()
#             mask_crop_cpu = mask_crop.cpu().squeeze(0).numpy().astype(np.uint8)
#
#             img_pil = T.ToPILImage()(img_crop_cpu)
#             mask_pil = Image.fromarray(mask_crop_cpu, mode='L')
#
#             # 生成新的文件名
#             base_name = os.path.splitext(img_name)[0]
#             crop_img_name = f"{base_name}_crop_{idx}.png"
#             crop_mask_name = f"{os.path.splitext(mask_name)[0]}_crop_{idx}.png"
#
#             # 保存裁剪后的图像和掩码
#             img_pil.save(os.path.join(output_images_dir, crop_img_name))
#             mask_pil.save(os.path.join(output_masks_dir, crop_mask_name))
#
#
# if __name__ == "__main__":
#     # 参数设置
#     ORIGINAL_IMAGES_DIR = 'D:\sea\\test_imgs'  # 原图文件夹路径
#     ORIGINAL_MASKS_DIR = 'D:\sea\\test_labels'  # 标签文件夹路径
#     OUTPUT_IMAGES_DIR = 'D:\sea\cropped_imgs'  # 裁剪后图像保存路径
#     OUTPUT_MASKS_DIR = 'D:\sea\cropped_labels'  # 裁剪后标签保存路径
#     CROP_SIZE = 5000  # 裁剪大小 n x n
#     STRIDE = 5000  # 步长，控制重叠程度
#     NUM_CLASSES = 2  # 类别数量，根据你的数据集设置
#     MAX_RATIO = 0.8  # 最大类别比例
#
#     # 检查是否有可用的 GPU
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"使用设备: {device}")
#
#     process_dataset(
#         images_dir=ORIGINAL_IMAGES_DIR,
#         masks_dir=ORIGINAL_MASKS_DIR,
#         output_images_dir=OUTPUT_IMAGES_DIR,
#         output_masks_dir=OUTPUT_MASKS_DIR,
#         crop_size=CROP_SIZE,
#         stride=STRIDE,
#         num_classes=NUM_CLASSES,
#         max_ratio=MAX_RATIO,
#         device=device
#     )
